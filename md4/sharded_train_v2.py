import functools
from collections.abc import Callable, Mapping
from typing import Any

import flax.linen as nn
import grain.python as grain
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers, metrics, periodic_actions
from etils import epath
from flax.training import orbax_utils
from jax.experimental import checkify
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from md4 import (
    input_pipeline,
    sampling,
)
from md4.utils import (
    checkpoint_utils,
    partial_load_utils,
    rdkit_utils,
    state_utils,
    utils,
    wandb_writer,
)

# Set up device mesh for FSDP sharding
device_count = len(jax.devices())
logging.info(f"Setting up FSDP mesh with {device_count} devices")

# Create mesh based on device count
if device_count >= 8:
    # Use 2D mesh for both data and model parallelism
    mesh_shape = (2, device_count // 2)  # e.g., (2, 4) for 8 devices
    mesh = jax.make_mesh(mesh_shape, ("data", "model"))
elif device_count >= 4:
    # Use 1D mesh for FSDP only
    mesh = jax.make_mesh((device_count,), ("model",))
else:
    # Fallback to single device or small device count
    mesh = jax.make_mesh((device_count,), ("model",))

logging.info(f"Created mesh: {mesh}")


def mesh_sharding(pspec: P) -> NamedSharding:
    """Helper function to create NamedSharding from PartitionSpec."""
    return NamedSharding(mesh, pspec)


def merge_batch_stats(
    train_state: state_utils.TrainState,
) -> state_utils.TrainState:
    """Merge model batch stats across devices for FSDP."""
    if jax.tree.leaves(train_state.state):
        # For FSDP, batch stats should already be synchronized
        # No explicit merge needed as we're using jit instead of pmap
        return train_state
    else:
        return train_state


def cosine_decay(
    lr: Any, current_step: Any, total_steps: Any
) -> Any:  # pytype: disable=invalid-annotation
    """Cosine decay that accepts Python scalars or JAX arrays."""
    current_step = jnp.asarray(current_step, dtype=jnp.float32)
    total_steps = jnp.maximum(1.0, jnp.asarray(total_steps, dtype=jnp.float32))
    lr = jnp.asarray(lr, dtype=jnp.float32)
    ratio = jnp.maximum(0.0, current_step / total_steps)
    mult = 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
    return mult * lr  # pytype: disable=bad-return-type  # jax-types


def get_learning_rate(
    step: int,
    *,
    base_learning_rate: float,
    num_steps: int,
    warmup_steps: int | None = None,
    schedule_type: str = "cosine",
) -> Any:  # pytype: disable=invalid-annotation
    """Learning rate schedule helper.

    Supports:
      - "cosine": single cosine decay with (optional) warmup.
      - "constant": constant LR after warmup.
      - "cyclic_cosine": cosine annealing with warm restarts (cyclic cosine).

    The cyclic variant optionally allows inline parameter overrides encoded in
    the schedule_type string, e.g.:
        "cyclic_cosine;cycle_length=1000;min_lr=1e-5"
    Parameters (all optional) for cyclic_cosine:
        cycle_length:   Number of steps per cycle (default: num_steps // 10, >=1).
        min_lr:         Minimum LR at the end of each cycle (default: 0.0).
        decay_factor:   Multiplicative factor applied to base LR after each cycle
                        restart (default: 1.0, i.e., no decay of peaks).
    Warmup (if warmup_steps provided) is applied only to the very beginning of
    training (first warmup_steps) scaling the scheduled LR linearly.
    """
    logging.info(
        "get_learning_rate(step=%s, base_learning_rate=%s, num_steps=%s, schedule_type=%s)",
        step,
        base_learning_rate,
        num_steps,
        schedule_type,
    )

    # Handle warmup (gracefully if warmup_steps is None or 0).
    if warmup_steps is None or warmup_steps <= 0:
        warmup = 1.0
        effective_step = step
        effective_total = num_steps
    else:
        warmup = jnp.minimum(1.0, step / warmup_steps)
        effective_step = jnp.maximum(0, step - warmup_steps)
        effective_total = jnp.maximum(1, num_steps - warmup_steps)

    # Allow parameter overrides for cyclic cosine via semi-colon separated kv pairs.
    schedule_base = schedule_type.split(";")[0]
    extra_params = schedule_type.split(";")[1:]
    parsed: dict[str, str] = {}
    for kv in extra_params:
        if "=" in kv:
            k, v = kv.split("=", 1)
            parsed[k.strip()] = v.strip()

    if schedule_base == "cosine":
        lr = cosine_decay(base_learning_rate, effective_step, effective_total)
    elif schedule_base == "constant":
        lr = base_learning_rate
    elif schedule_base == "cyclic_cosine":
        # Derive cycle_length (at least 1) and other params.
        default_cycle = max(1, num_steps // 10)
        try:
            cycle_length = int(parsed.get("cycle_length", default_cycle))
        except ValueError:  # Fall back to default if parsing fails.
            cycle_length = default_cycle
        cycle_length = max(1, cycle_length)

        try:
            min_lr = float(parsed.get("min_lr", 0.0))
        except ValueError:
            min_lr = 0.0
        try:
            decay_factor = float(parsed.get("decay_factor", 1.0))
        except ValueError:
            decay_factor = 1.0

        # Position within current cycle (after warmup region).
        cycle_index = jnp.floor_divide(effective_step, cycle_length)
        pos_in_cycle = jnp.mod(effective_step, cycle_length)

        # Optionally decay the peak LR each cycle.
        peak_lr = base_learning_rate * (decay_factor**cycle_index)

        # Cosine within the cycle from peak_lr down to min_lr.
        cosine_ratio = pos_in_cycle / cycle_length
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * cosine_ratio))
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_type}")

    return jnp.asarray(
        lr * warmup, dtype=jnp.float32
    )  # pytype: disable=bad-return-type  # jax-types


def loss_fn(params, state, rng, model, batch, train=False):
    """Loss function."""
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if train:
        _, dropout_rng = jax.random.split(rng)
        rngs["dropout"] = dropout_rng

    variables = {"params": params, **state}
    if "image" in batch:
        x = batch["image"]
    elif "text" in batch:
        x = batch["text"]
    elif "smiles" in batch:
        x = batch["smiles"]
    else:
        raise ValueError("Unsupported targets/tasks.")

    # Get model dtype for proper mixed precision handling
    conditioning = state_utils.get_conditioning_from_batch(batch, dtype=jnp.float32)

    new_state = {}
    if train:
        metrics_dict, new_state = model.apply(
            variables,
            x,
            cond=conditioning,
            train=train,
            rngs=rngs,
            mutable=list(state.keys()),
        )
    else:
        metrics_dict = model.apply(
            variables, x, cond=conditioning, train=train, rngs=rngs
        )

    loss = metrics_dict["loss"]
    if train:
        return loss, (new_state, metrics_dict)
    return loss, metrics_dict


@jax.jit
def merge_metrics(a_tree, b_tree):
    return jax.tree.map(lambda a, b: a + b, a_tree, b_tree)


def train_step(
    train_state: state_utils.TrainState,
    batch: Mapping[str, jnp.ndarray],
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    learning_rate_fn: Callable[[int], float],
    train_metrics_class: metrics.Collection,
    ema_rate: float = 0.0,
    num_microbatches: int | None = None,
) -> tuple[state_utils.TrainState, metrics.Collection]:
    """Perform a single training step with FSDP sharding."""
    logging.info("train_step(batch=%s)", batch)
    rng, new_rng = jax.random.split(train_state.rng)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    if num_microbatches is None or num_microbatches <= 1:
        (_, (new_state, metrics_dict)), grads = grad_fn(
            train_state.params, train_state.state, rng, model, batch, train=True
        )
    else:
        batch_size = next(iter(batch.values())).shape[0]
        assert batch_size % num_microbatches == 0, (
            "Batch size isn't divided evenly by num_microbatches."
        )
        microbatch_size = batch_size // num_microbatches
        logging.info(
            "using microbatches: %d microbatches, %d size",
            num_microbatches,
            microbatch_size,
        )

        def get_microbatch(
            batch: Mapping[str, jnp.ndarray], idx: int
        ) -> Mapping[str, jnp.ndarray]:
            """Fetch microbatch slice from possibly-packed input data."""
            offset = idx * microbatch_size
            length = microbatch_size
            starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
            limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
            return {
                k: jax.lax.dynamic_slice(b, starts[k], limits[k])
                for k, b in batch.items()
            }

        def metrics_and_grad(loop_cnt, rng, train_state_state):
            _, mbrng = jax.random.split(rng)
            mb = get_microbatch(batch, loop_cnt)

            (_, (new_state, metrics_dict)), grads = grad_fn(
                train_state.params, train_state_state, mbrng, model, mb, train=True
            )
            return metrics_dict, grads, new_state

        def per_microbatch_train_step(loop_cnt, carry):
            (rng, grad_accum, prev_metrics_dict, train_state_state) = carry
            metrics_dict, grads, train_state_state = metrics_and_grad(
                loop_cnt, rng, train_state_state
            )

            grad_accum = jax.tree.map(jnp.add, grad_accum, grads)
            metrics_dict = jax.lax.cond(
                loop_cnt == 0,
                lambda _: metrics_dict,
                lambda _: merge_metrics(prev_metrics_dict, metrics_dict),
                None,
            )
            return rng, grad_accum, metrics_dict, train_state_state

        # Initialize gradient accumulation loop state.
        accum_dtype = jnp.float32
        grad_accum_init = jax.tree.map(
            lambda x: jnp.zeros(x.shape, accum_dtype), train_state.params
        )
        initial_metrics_shape, _, _ = jax.eval_shape(
            metrics_and_grad,
            loop_cnt=0,
            rng=rng,
            train_state_state=train_state.state,
        )

        initial_metrics = {
            k: jnp.zeros(shape=v.shape, dtype=v.dtype)
            for k, v in initial_metrics_shape.items()
        }

        loop_init = (
            rng,
            grad_accum_init,
            initial_metrics,
            train_state.state,
        )
        _, grads, metrics_dict, train_state_state = jax.lax.fori_loop(
            0, num_microbatches, per_microbatch_train_step, loop_init
        )
        metrics_dict = jax.tree.map(lambda x: x / num_microbatches, metrics_dict)
        new_state = train_state_state

    # With FSDP and jit, gradients are already properly reduced
    # No explicit pmean needed as we're using data parallelism through sharding
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
    if ema_rate > 0.0:
        new_ema_params = jax.tree_util.tree_map(
            lambda x, y: x + (1.0 - ema_rate) * (y - x),
            train_state.ema_params,
            new_params,
        )
    else:
        new_ema_params = None

    new_train_state = train_state.replace(
        step=train_state.step + 1,
        rng=new_rng,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state,
        state=new_state,
    )

    metrics_update = train_metrics_class.single_from_model_output(
        learning_rate=learning_rate_fn(train_state.step),
        **metrics_dict,
    )
    return new_train_state, metrics_update


def eval_step(
    model: nn.Module,
    rng: jax.Array,
    train_state: state_utils.TrainState,
    batch: Mapping[str, jax.Array],
    eval_metrics_class: metrics.Collection,
    ema_rate: float = 0.0,
) -> metrics.Collection:
    """Compute the metrics for the given model in inference mode with FSDP."""
    logging.info("eval_step(batch=%s)", batch)
    axis_index = jax.lax.axis_index("data")
    rng = jax.random.fold_in(rng, axis_index)
    params = train_state.ema_params if ema_rate > 0.0 else train_state.params

    _, metrics_dict = loss_fn(params, train_state.state, rng, model, batch, train=False)
    return eval_metrics_class.single_from_model_output(
        learning_rate=0.0, **metrics_dict
    )


def evaluate(
    jit_eval_step: Any,
    rng: jax.Array,
    train_state: state_utils.TrainState,
    eval_loader: grain.DataLoader,
    num_eval_steps: int = -1,
    data_sharding: NamedSharding = None,
):
    """Evaluate the model on the given dataset (legacy pmap version)."""
    logging.info("Starting evaluation.")
    eval_metrics = None
    with utils.StepTraceContextHelper("eval", 0) as trace_context:
        # Use `iter` to reset the eval_loader before each evaluation.
        for step, batch_raw in enumerate(iter(eval_loader)):
            rng, sub_rng = jax.random.split(rng)
            # Apply sharding to batch
            batch = jax.tree.map(
                lambda x: jax.device_put(x, data_sharding) if data_sharding else x,
                batch_raw,
            )

            metrics_update = jit_eval_step(
                sub_rng, train_state, batch
            )
            eval_metrics = (
                metrics_update
                if eval_metrics is None
                else eval_metrics.merge(metrics_update)
            )
            if num_eval_steps > 0 and step + 1 == num_eval_steps:
                break
            trace_context.next_step()

    if eval_metrics is None:
        raise ValueError(f"Eval dataset {eval_loader} was empty.")
    return eval_metrics


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
      olddir: Optional directory to load old checkpoints from for partial loading.
        If provided, checkpoints will be loaded from olddir but saved to workdir.
    """
    workdir = epath.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rng = utils.get_rng(config.seed)
    logging.info("Using random seed %s.", rng)
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Start the profiler if requested
    if config.get("start_profiler", False) and jax.process_index() == 0:
        logging.info("Starting profiler.")
        jax.profiler.start_server(9999)

    # Add wandb writer to the multi-writer if we're on the main process
    if (
        jax.process_index() == 0
        and hasattr(writer, "_writers")
        and config.get("enable_wandb", False)
    ):
        wandb_w = wandb_writer.WandBWriter(
            project=config.get("wandb_project", "md4"), **config.get("wandb_kwargs", {})
        )
        writer._writers = tuple([wandb_w] + list(writer._writers))
        logging.info("Added WandB writer to metric writers.")

    # Learning rate schedule.
    assert config.batch_size % jax.device_count() == 0
    per_device_batch_size = config.batch_size // jax.device_count()
    num_train_steps = input_pipeline.get_num_train_steps(config)
    steps_per_epoch = num_train_steps // config.num_epochs
    logging.info(
        "num_train_steps=%d, steps_per_epoch=%d", num_train_steps, steps_per_epoch
    )
    schedule_fn = functools.partial(
        get_learning_rate,
        base_learning_rate=config.learning_rate,
        num_steps=num_train_steps,
        warmup_steps=config.warmup_steps,
        schedule_type=config.learning_rate_schedule,
    )

    # Build input pipeline.
    rng, data_seed = jax.random.split(rng)
    data_seed = int(
        jax.random.randint(data_seed, [], minval=0, maxval=np.iinfo(np.int32).max)
    )
    # The input pipeline runs on each process and loads data for local TPUs.
    train_loader, eval_loaders, dataset_info = input_pipeline.create_datasets(
        config, data_seed
    )

    # Determine if we need to create an iterator based on loader type
    # Grain loaders are DataLoader instances and need iter(), while TF iterators are already iterators
    is_grain_loader = isinstance(train_loader, grain.DataLoader)
    train_iter = iter(train_loader) if is_grain_loader else train_loader

    # Initialize sharding
    data_sharding = mesh_sharding(P("data", None))  # Shard data across data axis
    replicated_sharding = mesh_sharding(P())
    state_sharding = None

    # Initialize model.
    rng, model_rng = jax.random.split(rng)
    data_shape = input_pipeline.get_data_shape(config)

    if config.get("frozen", False):
        logging.error("Not implemented for fsdp")
        raise NotImplementedError()
    else:
        logging.info("Creating train state.")
        model, optimizer, train_state, metrics_class, state_sharding = (
            state_utils.create_sharded_train_state(
                config,
                data_sharding,
                replicated_sharding,
                mesh,
                model_rng,
                input_shape=(per_device_batch_size,) + data_shape,
                schedule_fn=schedule_fn,
            )
        )

    # Set up checkpointing of the model and the input pipeline.
    # Get both save and load checkpoint managers
    save_checkpoint_manager, load_checkpoint_manager = (
        checkpoint_utils.get_checkpoint_managers(
            config, workdir, olddir, is_grain_loader=is_grain_loader
        )
    )

    # Retrieve data from previous checkpoints if possible.
    if load_checkpoint_manager.latest_step() is not None:
        # Check if we should use partial loading
        if partial_load_utils.should_use_partial_loading(config):
            logging.error("Partial Loading is not supported for fsdp")
            raise NotImplementedError()
        else:
            # Standard checkpoint loading
            train_state, _ = partial_load_utils.standard_checkpoint_loading(
                train_state=train_state,
                train_iter=train_iter if is_grain_loader else None,
                checkpoint_manager=load_checkpoint_manager,
            )

    logging.info("Batch Size: %s", config.batch_size)

    # Place train_state with appropriate sharding
    train_state = jax.device_put(train_state, state_sharding)

    # JIT compile training and eval functions with sharding
    train_step_func = functools.partial(
        train_step,
        model=model,
        optimizer=optimizer,
        train_metrics_class=metrics_class,
        learning_rate_fn=schedule_fn,
        ema_rate=config.ema_rate,
        num_microbatches=config.get("num_microbatches", None),
    )

    if config.check_nans:
        train_step_func = checkify.checkify(
            train_step_func, errors=checkify.float_checks
        )

    # Use jit with sharding instead of pmap
    jit_train_step = jax.jit(
        train_step_func,
        in_shardings=(state_sharding, data_sharding),  # Input shardings
        out_shardings=(state_sharding, None),  # Output shardings
        donate_argnames=("train_state",),  # Donate train_state for memory efficiency
    )

    jit_eval_step = jax.jit(
        functools.partial(
            eval_step,
            model=model,
            eval_metrics_class=metrics_class,
            ema_rate=config.ema_rate,
        ),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
        ]
    train_metrics = None

    # With FSDP/jit, we don't need to unreplicate
    initial_step = int(train_state.step)
    eval_metrics_cpu = None

    # Run training within mesh context
    with mesh:
        with metric_writers.ensure_flushes(writer):
            # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
            for step in range(initial_step + 1, num_train_steps + 1):
                is_last_step = step == num_train_steps

                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    batch_raw = next(train_iter)
                    # Apply sharding to batch data
                    batch = jax.tree.map(
                        lambda x: jax.device_put(x, data_sharding), batch_raw
                    )

                    if config.check_nans:
                        errs, (train_state, metrics_update) = jit_train_step(
                            train_state, batch
                        )
                        errs.throw()
                    else:
                        train_state, metrics_update = jit_train_step(
                            train_state, batch
                        )

                    train_metrics = (
                        metrics_update
                        if train_metrics is None
                        else train_metrics.merge(metrics_update)
                    )

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            if step % config.log_loss_every_steps == 0 or is_last_step:
                writer.write_scalars(step, train_metrics.compute())
                train_metrics = None

                if step == 1 or step % config.eval_every_steps == 0 or is_last_step:
                    for split, eval_loader in eval_loaders.items():
                        rng, eval_rng = jax.random.split(rng)
                        with report_progress.timed("eval"):
                            train_state = merge_batch_stats(train_state)
                            # Pass jit_eval_step instead of p_eval_step
                            eval_metrics = evaluate(
                                jit_eval_step,
                                eval_rng,
                                train_state,
                                eval_loader,
                                config.num_eval_steps,
                                data_sharding,
                            )
                        eval_metrics_cpu = jax.tree_util.tree_map(
                            np.array, eval_metrics.compute()
                        )
                        eval_metrics_cpu = {
                            split + "_" + k: v for k, v in eval_metrics_cpu.items()
                        }
                        writer.write_scalars(step, eval_metrics_cpu)

                # if hasattr(model, "sample_step"):
                #     with report_progress.timed("sample"):
                #         _, sample_rng = jax.random.split(rng)
                #         dummy_loader = train_loader
                #         dummy_batch_raw = next(iter(dummy_loader))
                #         dummy_batch = jax.tree.map(
                #             lambda x: jax.device_put(x, data_sharding),
                #             dummy_batch_raw
                #         )
                #         dummy_inputs = (
                #             dummy_batch[config.task_type]
                #             if "smiles" not in dummy_batch
                #             else dummy_batch["smiles"]
                #         )
                #         # Get model dtype for proper mixed precision handling
                #         model_dtype = getattr(model, 'dtype', jnp.float32)
                #         conditioning = state_utils.get_conditioning_from_batch(
                #             dummy_batch, dtype=model_dtype
                #         )

                #         # For FSDP, we need to adapt the sampling
                #         samples = sampling.generate(
                #             model,
                #             train_state,
                #             sample_rng,
                #             dummy_inputs,
                #             conditioning,
                #             False,
                #         )

                #         # With FSDP, samples are already gathered
                #         all_samples = samples
                #         if hasattr(all_samples, 'shape'):
                #             all_samples = all_samples.reshape(-1, *data_shape)
                #         else:
                #             all_samples = jnp.array(all_samples).reshape(-1, *data_shape)
                #         if config.task_type == "image":
                #             sample_grid = utils.generate_image_grids(all_samples)
                #             writer.write_images(step, {"samples": sample_grid})
                #             del all_samples, sample_grid
                #         elif config.task_type == "text":
                #             tokenizer = dataset_info["tokenizer"]
                #             texts = None
                #             try:
                #                 texts = tokenizer.batch_decode(
                #                     all_samples,
                #                     skip_special_tokens=False,
                #                     clean_up_tokenization_spaces=True,
                #                 )
                #                 # writer.write_texts(step, {"samples": texts})

                #                 # Calculate SMILES validity for pubchem_large dataset
                #                 if (
                #                     config.dataset
                #                     in [
                #                         "pubchem_large",
                #                         "msg_finetune",
                #                         "pubchem_large_text",
                #                     ]
                #                     and texts is not None
                #                 ):
                #                     validity_metrics = (
                #                         rdkit_utils.calculate_smiles_validity(texts)
                #                     )
                #                     # Write validity metrics to the writer
                #                     validity_scalars = {
                #                         f"sample_{k}": v
                #                         for k, v in validity_metrics.items()
                #                     }
                #                     writer.write_scalars(step, validity_scalars)
                #             except Exception as e:
                #                 logging.error("Error decoding texts: %s", e)

                if step % config.checkpoint_every_steps == 0 or is_last_step:
                    with report_progress.timed("checkpoint"):
                        # Prepare checkpoint items - with FSDP, no unreplication needed
                        checkpoint_items = dict(
                            train_state=train_state,
                        )


                        # Only include train_iter for grain loaders
                        if is_grain_loader:
                            checkpoint_items["train_iter"] = train_iter

                        save_args = orbax_utils.save_args_from_target(checkpoint_items)
                        save_checkpoint_manager.save(
                            step,
                            items=checkpoint_items,
                            metrics=jax.tree_util.tree_map(
                                lambda x: x.item() if hasattr(x, "item") else x,
                                eval_metrics_cpu
                                if eval_metrics_cpu is not None
                                else {},
                            ),
                            args=save_args,
                        )

    logging.info("Finishing training at step %d", num_train_steps)
