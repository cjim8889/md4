# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Methods for training MD4/GenMD4 on text/image datasets."""

from collections.abc import Callable, Mapping, Sequence
import copy
import functools
from typing import Any

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
from etils import epath
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import grain.python as grain
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from orbax import checkpoint as orbax_checkpoint
from flax import traverse_util

from md4 import input_pipeline
from md4 import input_pipeline_v2
from md4 import rdkit_utils
from md4 import sampling
from md4 import utils
from md4.models import utils as model_utils
from md4 import partial_load_utils
from md4 import state_utils

def merge_batch_stats(replicated_state: state_utils.TrainState) -> state_utils.TrainState:
    """Merge model batch stats."""
    if jax.tree.leaves(replicated_state.state):
        cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "batch"), "batch")
        return replicated_state.replace(
            state=cross_replica_mean(replicated_state.state)
        )
    else:
        return replicated_state


def _get_checkpoint_manager(
    config: ml_collections.ConfigDict, workdir: epath.PathLike, create: bool = True
) -> orbax_checkpoint.CheckpointManager:
    """Loads the orbax checkpoint manager for train state and data iterator."""
    # The keys in this dict should match the keys in `checkpointed_state`.
    checkpointers = dict(
        train_state=orbax_checkpoint.PyTreeCheckpointer(),
        train_iter=orbax_checkpoint.Checkpointer(
            grain.PyGrainCheckpointHandler()
        ),  # pytype:disable=wrong-arg-types
    )
    checkpoint_dir = epath.Path(workdir) / "checkpoints"
    # keep_period = (
    #     config.checkpoint_keep_period if config.checkpoint_keep_period > 0 else None
    # )
    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointers=checkpointers,
        options=orbax_checkpoint.CheckpointManagerOptions(
            create=create,
            # preservation_policy=orbax_checkpoint.checkpoint_managers.LatestN(n=20),
            best_fn=lambda x: x["validation_loss"]
            if "validation_loss" in x
            else x["loss"],
            best_mode="min",
            max_to_keep=20,
        ),
    )


def cosine_decay(lr: float, current_step: float, total_steps: float) -> float:
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
) -> float:
    """Cosine learning rate schedule."""
    logging.info(
        "get_learning_rate(step=%s, base_learning_rate=%s, num_steps=%s",
        step,
        base_learning_rate,
        num_steps,
    )
    warmup = jnp.minimum(1.0, step / warmup_steps)
    if schedule_type == "cosine":
        lr = cosine_decay(
            base_learning_rate, step - warmup_steps, num_steps - warmup_steps
        )
    elif schedule_type == "constant":
        lr = base_learning_rate
    else:
        raise NotImplementedError()
    return lr * warmup  # pytype: disable=bad-return-type  # jax-types


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

    conditioning = state_utils.get_conditioning_from_batch(batch)

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
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_state: state_utils.TrainState,
    batch: Mapping[str, jnp.ndarray],
    learning_rate_fn: Callable[[int], float],
    train_metrics_class: Any,
    ema_rate: float = 0.0,
    num_microbatches: int | None = None,
) -> tuple[state_utils.TrainState, metrics.Collection]:
    """Perform a single training step."""
    logging.info("train_step(batch=%s)", batch)
    rng, new_rng = jax.random.split(train_state.rng)
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

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

    # Compute average gradient across multiple workers.
    grads = jax.lax.pmean(grads, axis_name="batch")
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

    metrics_update = train_metrics_class.gather_from_model_output(
        learning_rate=learning_rate_fn(train_state.step),
        **metrics_dict,
    )
    return new_train_state, metrics_update


def eval_step(
    model: nn.Module,
    rng: jnp.ndarray,
    train_state: state_utils.TrainState,
    batch: Mapping[str, jnp.ndarray],
    eval_metrics_class: Any,
    ema_rate: float = 0.0,
) -> metrics.Collection:
    """Compute the metrics for the given model in inference mode."""
    logging.info("eval_step(batch=%s)", batch)
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    params = train_state.ema_params if ema_rate > 0.0 else train_state.params

    _, metrics_dict = loss_fn(params, train_state.state, rng, model, batch, train=False)
    return eval_metrics_class.gather_from_model_output(
        learning_rate=0.0, **metrics_dict
    )


def evaluate(
    p_eval_step: Any,
    rng: jnp.ndarray,
    train_state: state_utils.TrainState,
    eval_loader: grain.DataLoader,
    num_eval_steps: int = -1,
):
    """Evaluate the model on the given dataset."""
    logging.info("Starting evaluation.")
    eval_metrics = None
    with utils.StepTraceContextHelper("eval", 0) as trace_context:
        # Use `iter` to reset the eval_loader before each evaluation.
        for step, batch in enumerate(iter(eval_loader)):
            rng, sub_rng = jax.random.split(rng)
            sub_rng = flax_utils.replicate(sub_rng)
            batch = utils.reshape_batch(batch)
            metrics_update = flax_utils.unreplicate(
                p_eval_step(rng=sub_rng, train_state=train_state, batch=batch)
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
    create_datasets = (
        input_pipeline_v2.create_datasets
        if config.get("use_v2_input_pipeline", None)
        else input_pipeline.create_datasets
    )
    train_loader, eval_loaders, dataset_info = create_datasets(config, data_seed)
    train_iter = (
        iter(train_loader)
        if config.dataset not in ["pubchem_large", "msg_finetune"]
        else train_loader
    )

    # Initialize model.
    rng, model_rng = jax.random.split(rng)
    data_shape = input_pipeline.get_data_shape(config)

    if config.get("frozen", False):
        logging.info("Creating frozen train state.")
        model, optimizer, train_state, metrics_class = state_utils.create_train_state(  # pylint: disable=invalid-name
            config,
            model_rng,
            input_shape=(per_device_batch_size,) + data_shape,
            schedule_fn=schedule_fn,
        )
    else:
        logging.info("Creating train state.")
        model, optimizer, train_state, metrics_class = state_utils.create_train_state(
            config,
            model_rng,
            input_shape=(per_device_batch_size,) + data_shape,
            schedule_fn=schedule_fn,
        )

    # Set up checkpointing of the model and the input pipeline.
    # Create checkpoint manager for saving (new checkpoints)
    save_checkpoint_manager = _get_checkpoint_manager(config, workdir, create=True)

    # Create checkpoint manager for loading (old checkpoints if olddir is provided)
    if olddir is not None:
        load_checkpoint_manager = _get_checkpoint_manager(config, olddir, create=False)
    else:
        load_checkpoint_manager = save_checkpoint_manager

    # Retrieve data from previous checkpoints if possible.
    if load_checkpoint_manager.latest_step() is not None:
        # Check if we should use partial loading
        if partial_load_utils.should_use_partial_loading(config):
            try:
                train_state, _ = partial_load_utils.partial_load_checkpoint(
                    config=config,
                    train_state=train_state,
                    train_iter=train_iter
                    if config.dataset not in ["pubchem_large", "msg_finetune"]
                    else None,
                    checkpoint_manager=load_checkpoint_manager,
                    create_train_state_fn=state_utils.create_train_state,
                    schedule_fn=schedule_fn,
                    per_device_batch_size=per_device_batch_size,
                    data_shape=data_shape,
                )
            except Exception as e:
                logging.error(f"Partial loading failed: {e}")
                raise
        else:
            # Standard checkpoint loading
            train_state, _ = partial_load_utils.standard_checkpoint_loading(
                train_state=train_state,
                train_iter=train_iter
                if config.dataset not in ["pubchem_large", "msg_finetune"]
                else None,
                checkpoint_manager=load_checkpoint_manager,
            )

    logging.info("Batch Size: %s", config.batch_size)

    # Distribute training.
    train_state = flax_utils.replicate(train_state)
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
    p_train_step = jax.pmap(train_step_func, axis_name="batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            model=model,
            eval_metrics_class=metrics_class,
            ema_rate=config.ema_rate,
        ),
        axis_name="batch",
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

    # Unreplicating from TPU is costly, so we only do it once at the start.
    initial_step = int(flax.jax_utils.unreplicate(train_state.step))
    eval_metrics_cpu = None
    with metric_writers.ensure_flushes(writer):
        # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
        for step in range(initial_step + 1, num_train_steps + 1):
            is_last_step = step == num_train_steps

            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = utils.reshape_batch(next(train_iter))
                if config.check_nans:
                    errs, (train_state, metrics_update) = p_train_step(
                        train_state=train_state, batch=batch
                    )
                    errs.throw()
                else:
                    train_state, metrics_update = p_train_step(
                        train_state=train_state, batch=batch
                    )
                metric_update = flax_utils.unreplicate(metrics_update)
                train_metrics = (
                    metric_update
                    if train_metrics is None
                    else train_metrics.merge(metric_update)
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
                        eval_metrics = evaluate(
                            p_eval_step,
                            eval_rng,
                            train_state,
                            eval_loader,
                            config.num_eval_steps,
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
                        dummy_batch = utils.reshape_batch(next(iter(dummy_loader)))
                        dummy_inputs = (
                            dummy_batch[config.task_type]
                            if "smiles" not in dummy_batch
                            else dummy_batch["smiles"]
                        )
                        conditioning = state_utils.get_conditioning_from_batch(dummy_batch)

                        samples = sampling.generate(
                            model,
                            train_state,
                            flax_utils.replicate(sample_rng),
                            dummy_inputs,
                            conditioning=conditioning,
                        )

                        all_samples = jax.pmap(
                            lambda x: jax.lax.all_gather(x, "batch"), axis_name="batch"
                        )(samples)
                        all_samples = flax_utils.unreplicate(all_samples)
                        all_samples = all_samples.reshape(-1, *data_shape)
                        if config.task_type == "image":
                            sample_grid = utils.generate_image_grids(all_samples)
                            writer.write_images(step, {"samples": sample_grid})
                            del all_samples, sample_grid
                        elif config.task_type == "text":
                            tokenizer = dataset_info["tokenizer"]
                            texts = None
                            try:
                                texts = tokenizer.batch_decode(
                                    all_samples,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=True,
                                )
                                writer.write_texts(step, {"samples": texts})

                                # Calculate SMILES validity for pubchem_large dataset
                                if (
                                    config.dataset in ["pubchem_large", "msg_finetune"]
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

            if step % config.checkpoint_every_steps == 0 or is_last_step:
                with report_progress.timed("checkpoint"):
                    train_state = merge_batch_stats(train_state)
                    if config.dataset in ["pubchem_large", "msg_finetune"]:
                        save_checkpoint_manager.save(
                            step,
                            items=dict(
                                train_state=jax.tree_util.tree_map(
                                    np.array, flax_utils.unreplicate(train_state)
                                ),
                            ),
                            metrics=jax.tree_util.tree_map(
                                lambda x: x.item(), eval_metrics_cpu
                            ),
                        )
                    else:
                        save_checkpoint_manager.save(
                            step,
                            items=dict(
                                train_state=jax.tree_util.tree_map(
                                    np.array, flax_utils.unreplicate(train_state)
                                ),
                                train_iter=train_iter,
                            ),
                            metrics=jax.tree_util.tree_map(
                                lambda x: x.item(), eval_metrics_cpu
                            ),
                        )

    logging.info("Finishing training at step %d", num_train_steps)
