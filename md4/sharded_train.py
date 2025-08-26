import functools
from collections.abc import Callable, Mapping
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers, metrics, periodic_actions
from etils import epath
from jax.experimental import checkify, multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from orbax import checkpoint as orbax_checkpoint

from md4 import (
    input_pipeline,
    sampling,
)
from md4.utils import (
    checkpoint_utils,
    learning_rate,
    rdkit_utils,
    state_utils,
    utils,
    wandb_writer,
)


def create_device_mesh(config):
    """Create device mesh based on configuration."""
    mesh_config = config.mesh_config
    mesh = jax.make_mesh(mesh_config.mesh_shape, mesh_config.mesh_axis_names)
    logging.info(f"Created mesh: {mesh}")
    return mesh


def mesh_sharding(mesh, pspec: P) -> NamedSharding:
    """Helper function to create NamedSharding from PartitionSpec."""
    return NamedSharding(mesh, pspec)


def loss_fn(params, state, rng, batch, model, train=False):
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


def _reshape_to_microbatches(
    batch: Mapping[str, jnp.ndarray],
    num_microbatches: int,
) -> Mapping[str, jnp.ndarray]:
    """Reshape leading batch axis -> [M, m, ...] and constrain sharding to P(None,'data',...)."""
    if num_microbatches is None or num_microbatches <= 1:
        return batch

    B = next(iter(batch.values())).shape[0]
    assert B % num_microbatches == 0, (
        "Batch size must be divisible by num_microbatches."
    )
    m = B // num_microbatches

    def to_mb(x):
        x = x.reshape((num_microbatches, m) + x.shape[1:])
        # Keep microbatch axis replicated; inner batch axis remains sharded on 'data'
        pspec = P(None, "data", *([None] * (x.ndim - 2)))
        return jax.lax.with_sharding_constraint(x, pspec)

    return jax.tree.map(to_mb, batch)


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
    """Perform a single training step."""
    logging.info("train_step(batch=%s)", batch)
    rng, new_rng = jax.random.split(train_state.rng)
    loss_fn_train = functools.partial(loss_fn, model=model, train=True)
    grad_fn = jax.value_and_grad(loss_fn_train, has_aux=True)

    if num_microbatches is None or num_microbatches <= 1:
        (loss, (new_state, metrics_dict)), grads = grad_fn(
            train_state.params,
            train_state.state,
            rng,
            batch,
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

        # 1) Reshape once; no dynamic_slice on a sharded axis.
        batch_mb = _reshape_to_microbatches(batch, num_microbatches)

        # 2) Build zeroed templates (dtype/shape match grads & metrics) using eval_shape on one microbatch.
        mb0 = jax.tree.map(lambda x: x[0], batch_mb)
        # Shape-only trace; NOTE: model is not an argument here.
        (out_avals, _grads_avals) = jax.eval_shape(
            grad_fn, train_state.params, train_state.state, rng, mb0
        )
        loss_aval, (_state_aval, metrics_avals) = out_avals

        # FP32 gradient accumulator (same tree as params)
        grad_acc0 = jax.tree.map(
            lambda p: jnp.zeros(p.shape, jnp.float32), train_state.params
        )
        metrics0 = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), metrics_avals)
        loss0 = jnp.zeros(loss_aval.shape, loss_aval.dtype)

        def body(carry, mb):
            loop_rng, state, grad_acc, metrics_acc, loss_acc = carry
            loop_rng, mbrng = jax.random.split(loop_rng)
            (loss, (state_new, mdict)), grads_param_dtype = grad_fn(
                train_state.params, state, mbrng, mb
            )
            grad_acc = jax.tree.map(
                lambda a, g: a + g.astype(jnp.float32), grad_acc, grads_param_dtype
            )
            metrics_acc = jax.tree.map(lambda a, b: a + b, metrics_acc, mdict)
            loss_acc = loss_acc + loss
            return (loop_rng, state_new, grad_acc, metrics_acc, loss_acc), None

        (_, new_state, grad_sum_f32, metrics_sum, loss_sum), _ = jax.lax.scan(
            body,
            (rng, train_state.state, grad_acc0, metrics0, loss0),
            batch_mb,
            unroll=1,
        )

        inv = 1.0 / float(num_microbatches)
        _ = loss_sum * inv  # loss used for gradient computation
        metrics_dict = jax.tree.map(lambda x: x * inv, metrics_sum)
        # average grads, then cast back to param dtype for the optimizer step
        grads = jax.tree.map(
            lambda p, g: (g * inv).astype(p.dtype), train_state.params, grad_sum_f32
        )

    # 4) Optimizer + (optional) EMA
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)

    if ema_rate > 0.0:
        new_ema_params = jax.tree.map(
            lambda ema, p: ema + (1.0 - ema_rate) * (p - ema),
            train_state.ema_params,
            new_params,
        )
    else:
        new_ema_params = None

    new_train_state = train_state.replace(  # type: ignore[attr-defined]
        step=train_state.step + 1,
        rng=new_rng,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state,
        state=new_state,
    )

    new_metrics = train_metrics_class.single_from_model_output(
        learning_rate=learning_rate_fn(
            train_state.step
        ),  # lr at start of step (matches your original)
        **metrics_dict,
    )
    return new_train_state, new_metrics


def eval_step(
    rng: jax.Array,
    train_state: state_utils.TrainState,
    batch: Mapping[str, jax.Array],
    model: nn.Module,
    eval_metrics_class: metrics.Collection,
    ema_rate: float = 0.0,
) -> metrics.Collection:
    """Compute the metrics for the given model in inference mode with FSDP."""
    logging.info("eval_step(batch=%s)", batch)
    params = train_state.ema_params if ema_rate > 0.0 else train_state.params

    _, metrics_dict = loss_fn(params, train_state.state, rng, batch, model, train=False)
    return eval_metrics_class.single_from_model_output(
        learning_rate=0.0, **metrics_dict
    )


def _process_metrics(batch_metrics, matrics_class):
    batch_metrics = [jax.device_get(m) for m in batch_metrics]
    final_metrics = matrics_class.empty()
    for m in batch_metrics:
        final_metrics = final_metrics.merge(m)
    return final_metrics


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    eval_loss = metrics.Average.from_output("loss")
    eval_loss_diff = metrics.Average.from_output("loss_diff")
    eval_loss_prior = metrics.Average.from_output("loss_prior")
    eval_loss_recon = metrics.Average.from_output("loss_recon")


def make_global_array_data_only(
    data,  # numpy/jax array (global or host-local)
    global_shape: tuple[int, ...],
    data_sharding: NamedSharding | None,
    *,
    data_is_global: bool = True,  # set False if `data` is already host-sharded
) -> jax.Array:
    if data_sharding is None:
        return data

    return jax.make_array_from_process_local_data(data_sharding, data, global_shape)

    # # NOTE: Host-local is not actually implemented as it's more complex and not needed now.

    # pspec = data_sharding.spec
    # # --- Contract: only 'data' or None appear in the PartitionSpec ---
    # if any(ax not in (None, 'data') for ax in pspec):
    #     raise ValueError(f"Only data-axis sharding supported, got {pspec!r}")
    # if 'data' not in pspec:
    #     # nothing is sharded -> pure replication
    #     return jax.device_put(data, data_sharding)

    # data_dim = pspec.index('data')

    # # Easiest/safest path when every host sees the full global array:
    # if data_is_global:
    #     return jax.device_put(data, data_sharding)

    # # Host-local path: use the index map for this host's addressable devices.
    # idx_map = data_sharding.addressable_devices_indices_map(global_shape)

    # # Compute the offset (global -> host-local) along the data dimension.
    # starts = [idx[data_dim].start for idx in idx_map.values() if isinstance(idx[data_dim], slice)]
    # host_offset = min(starts) if starts else 0

    # local_arrays = []
    # for dev, gidx in idx_map.items():
    #     lidx = _shift_slice(gidx, data_dim, host_offset)
    #     shard = np.asarray(data[lidx])
    #     local_arrays.append(jax.device_put(shard, dev))

    # return jax.make_array_from_single_device_arrays(
    #     shape=global_shape,
    #     sharding=data_sharding,
    #     arrays=local_arrays,
    # )


def evaluate(
    jit_eval_step: Any,
    rng: jax.Array,
    train_state: state_utils.TrainState,
    eval_loader: Any,
    num_eval_steps: int = -1,
    data_sharding: NamedSharding | None = None,
    config: ml_collections.ConfigDict | None = None,
):
    """Evaluate the model on the given dataset (legacy pmap version)."""
    logging.info("Starting evaluation.")
    eval_metrics = []
    with utils.StepTraceContextHelper("eval", 0) as trace_context:
        # Use `iter` to reset the eval_loader before each evaluation.
        for step, batch_raw in enumerate(iter(eval_loader)):
            rng, sub_rng = jax.random.split(rng)

            batch = jax.tree.map(
                lambda x: make_global_array_data_only(
                    x,
                    global_shape=(config.batch_size, *x.shape[1:]),
                    data_sharding=data_sharding,
                    data_is_global=True,
                ),
                batch_raw,
            )

            metrics_update = jit_eval_step(sub_rng, train_state, batch)
            eval_metrics.append(metrics_update)
            if num_eval_steps > 0 and step + 1 == num_eval_steps:
                break
            trace_context.next_step()

    if eval_metrics is None:
        raise ValueError(f"Eval dataset {eval_loader} was empty.")
    eval_metrics = _process_metrics(eval_metrics, EvalMetrics)
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
    # Initialize multi-host JAX if enabled
    if config.get("initialize_multihost", False):
        jax.distributed.initialize()
        logging.info(
            f"JAX distributed initialized. Process {jax.process_index()}/{jax.process_count()}"
        )
        logging.info(f"Local devices: {jax.local_devices()}")
        logging.info(f"Global devices: {len(jax.devices())}")

    workdir = epath.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Set up device mesh from config
    mesh = create_device_mesh(config)

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
    num_train_steps = input_pipeline.get_num_train_steps(config)
    steps_per_epoch = num_train_steps // config.num_epochs
    logging.info(
        "num_train_steps=%d, steps_per_epoch=%d", num_train_steps, steps_per_epoch
    )
    schedule_fn = functools.partial(
        learning_rate.get_learning_rate,
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

    # Train loader is already an iterator from the input pipeline
    train_iter = train_loader

    # Initialize sharding
    data_sharding = mesh_sharding(mesh, P("data", None))  # Shard data across data axis
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
                mesh,
                model_rng,
                input_shape=(config.batch_size, *data_shape),
                schedule_fn=schedule_fn,
            )
        )

    # Set up checkpointing with preemption tolerance
    checkpoint_manager, load_checkpoint_manager = (
        checkpoint_utils.get_checkpoint_managers(config, workdir, olddir)
    )

    # Restore from checkpoint if available
    start_step = 0
    if load_checkpoint_manager.latest_step() is not None:
        start_step = load_checkpoint_manager.latest_step()
        logging.info(f"Restoring from step {start_step}")

        # Check if we should use partial loading
        if checkpoint_utils.should_use_partial_loading(config):
            logging.error("Partial Loading is not supported for fsdp")
            raise NotImplementedError()
        else:
            # Standard checkpoint loading
            train_state = checkpoint_utils.standard_checkpoint_loading(
                train_state=train_state,
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
            eval_metrics_class=EvalMetrics,
            ema_rate=config.ema_rate,
        ),
        in_shardings=(None, state_sharding, data_sharding),
        out_shardings=(None),
    )

    jit_generate = jax.jit(
        functools.partial(
            sampling.simple_generate,
            batch_size=config.batch_size,
            model=model,
            dummy_inputs=None,
            use_conditional_init=False,
        ),
        in_shardings=(None, state_sharding, data_sharding),
        out_shardings=(data_sharding),
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
    train_metrics = []

    # Use the restored step or start from 0
    initial_step = max(
        int(train_state.step), start_step if start_step is not None else 0
    )
    eval_metrics_cpu = None

    # Run training within mesh context
    with mesh:
        with metric_writers.ensure_flushes(writer):
            # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
            for step in range(initial_step + 1, num_train_steps + 1):
                is_last_step = step == num_train_steps

                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    batch_raw = next(train_iter)

                    batch = jax.tree.map(
                        lambda x: make_global_array_data_only(
                            x,
                            global_shape=(config.batch_size, *x.shape[1:]),
                            data_sharding=data_sharding,
                            data_is_global=True,
                        ),
                        batch_raw,
                    )

                    if config.check_nans:
                        errs, (train_state, metrics_update) = jit_train_step(
                            train_state, batch
                        )
                        errs.throw()
                    else:
                        train_state, metrics_update = jit_train_step(train_state, batch)

                    train_metrics.append(metrics_update)

                # Quick indication that training is happening.
                logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
                for h in hooks:
                    h(step)

                if step % config.log_loss_every_steps == 0 or is_last_step:
                    train_metrics = _process_metrics(train_metrics, metrics_class)
                    writer.write_scalars(step, train_metrics.compute())
                    train_metrics = []

                if step == 1 or step % config.eval_every_steps == 0 or is_last_step:
                    for split, eval_loader in eval_loaders.items():
                        rng, eval_rng = jax.random.split(rng)
                        with report_progress.timed("eval"):
                            eval_metrics = evaluate(
                                jit_eval_step,
                                eval_rng,
                                train_state,
                                eval_loader,
                                config.num_eval_steps,
                                data_sharding,
                                config,
                            )
                        eval_metrics_cpu = jax.tree_util.tree_map(
                            np.array, eval_metrics.compute()
                        )
                        eval_metrics_cpu = {
                            split + "_" + k: v for k, v in eval_metrics_cpu.items()
                        }
                        writer.write_scalars(step, eval_metrics_cpu)

                    if hasattr(model, "sample_step"):
                        with report_progress.timed("sample"):
                            _, sample_rng = jax.random.split(rng)
                            dummy_loader = train_loader
                            dummy_batch_raw = next(iter(dummy_loader))

                            dummy_batch = jax.tree.map(
                                lambda x: make_global_array_data_only(
                                    x,
                                    global_shape=(config.batch_size, *x.shape[1:]),
                                    data_sharding=data_sharding,
                                    data_is_global=True,
                                ),
                                dummy_batch_raw,
                            )

                            # Get model dtype for proper mixed precision handling
                            model_dtype = getattr(model, "dtype", jnp.float32)
                            conditioning = state_utils.get_conditioning_from_batch(
                                dummy_batch, dtype=model_dtype
                            )

                            samples = jit_generate(
                                sample_rng,
                                train_state,
                                conditioning,
                            )

                            # Get samples to CPU for decoding - handle multi-host properly
                            if config.get("initialize_multihost", False):
                                # In multi-host mode, use process_allgather to collect samples from all processes
                                all_samples = multihost_utils.process_allgather(samples)
                            else:
                                # Single-host: use regular device_get
                                all_samples = jax.device_get(samples)

                            # Only decode and calculate metrics on the main process to avoid redundant work
                            if config.task_type == "text" and jax.process_index() == 0:
                                tokenizer = dataset_info["tokenizer"]
                                texts = None
                                try:
                                    texts = tokenizer.batch_decode(
                                        all_samples,
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=True,
                                    )
                                    # writer.write_texts(step, {"samples": texts})

                                    # Calculate SMILES validity if enabled in config
                                    if (
                                        config.get("calculate_smiles_validity", False)
                                        and texts is not None
                                    ):
                                        validity_metrics = (
                                            rdkit_utils.calculate_smiles_validity(texts)
                                        )
                                        # Write validity metrics to the writer
                                        validity_scalars = {
                                            f"sample_{k}": v
                                            for k, v in validity_metrics.items()
                                        }
                                        writer.write_scalars(step, validity_scalars)
                                except Exception as e:
                                    logging.error("Error decoding texts: %s", e)

                # Save checkpoint with preemption tolerance
                with report_progress.timed("checkpoint"):
                    checkpoint_manager.save(
                        step,
                        args=orbax_checkpoint.args.StandardSave(train_state),
                        metrics=jax.tree_util.tree_map(
                            lambda x: x.item() if hasattr(x, "item") else x,
                            eval_metrics_cpu if eval_metrics_cpu is not None else {},
                        ),
                        force=is_last_step,
                    )

                # Check for preemption and handle gracefully
                if checkpoint_manager.reached_preemption(step):
                    logging.info(
                        f"Preemption detected at step {step}. Waiting for checkpointing to finish."
                    )
                    checkpoint_manager.wait_until_finished()
                    logging.info("Checkpointing completed. Exiting gracefully.")
                    return

    logging.info("Finishing training at step %d", num_train_steps)
