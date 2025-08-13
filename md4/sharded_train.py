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

"""Methods for training VD3 on image datasets."""

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
from flax import struct
import flax.linen as nn
import grain.python as grain
import jax
from jax.experimental import checkify
from jax.experimental import mesh_utils
import jax.numpy as jnp
# pylint: disable=g-importing-member

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import ml_collections
import numpy as np
import optax
from orbax import checkpoint as orbax_checkpoint

from md4 import input_pipeline
from md4 import multihost_dataloading
from md4 import sampling
from md4 import utils
from md4.models import utils as model_utils


class TrainState(struct.PyTreeNode):
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """

  rng: jnp.ndarray
  step: int
  params: Any
  ema_params: Any
  opt_state: optax.OptState
  state: Any
  tx: optax.GradientTransformation = struct.field(pytree_node=False)


class InferState(struct.PyTreeNode):
  """State of the model and the inference.

  This includes parameters only.
  """

  ema_params: Any
  state: Any


def _get_checkpoint_manager(
    workdir: epath.PathLike,
) -> orbax_checkpoint.CheckpointManager:
  # The keys in this dict should match the keys in `checkpointed_state`.
  checkpointers = dict(
      train_state=orbax_checkpoint.PyTreeCheckpointer(),
      train_iter=orbax_checkpoint.Checkpointer(
          grain.PyGrainCheckpointHandler()
      ),  # pytype:disable=wrong-arg-types
  )
  checkpoint_dir = epath.Path(workdir) / "checkpoints"
  return orbax_checkpoint.CheckpointManager(
      checkpoint_dir,
      checkpointers=checkpointers,
      options=orbax_checkpoint.CheckpointManagerOptions(
          max_to_keep=1, create=True
      ),
  )


def load_last_state(
    workdir: epath.PathLike, checkpointed_state: Mapping[str, Any] | None
) -> TrainState:
  """Loads the last state from Orbax.

  Args:
    workdir: The working directory to store Orbax state.
    checkpointed_state: an optional dictionary of object name ("train_state" and
      "train_iter") to restorable object. `None` to let Orbax restore from disk.

  Returns:
    The last checkpointed train state.
  """
  checkpoint_manager = _get_checkpoint_manager(workdir)
  if checkpoint_manager.latest_step() is None:
    raise ValueError("No last step found. Orbax has not run yet.")
  checkpointed_state = checkpoint_manager.restore(
      checkpoint_manager.latest_step(), items=checkpointed_state
  )
  return checkpointed_state["train_state"]


def create_train_metrics_class(config: ml_collections.ConfigDict):
  if config.model_type.endswith("diffusion"):
    metric_keys = sorted(
        ["loss_prior", "loss_diff", "loss_recon", "loss", "learning_rate"]
    )
  elif config.model_type == "ar":
    metric_keys = sorted(["loss", "learning_rate"])
  else:
    raise NotImplementedError()
  logging.info("metric_keys: %s", metric_keys)
  return create_train_metrics_class_from_keys(metric_keys)


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces.

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned leaves.

  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree.map(
      lambda x: x.unbox()
      if isinstance(x, flax.linen.spmd.LogicallyPartitioned)
      else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,
    input_shape: Sequence[int],
    schedule_fn: Callable[[Any], Any],
    model: nn.Module,
) -> TrainState:
  """Create train state."""
  if config.classes > 0:
    conditioning = jnp.zeros(input_shape[0], dtype="int32")
  else:
    conditioning = None
  rng, sample_rng, init_rng = jax.random.split(rng, 3)
  variables = model.init(
      {"sample": sample_rng, "params": init_rng},
      jnp.ones(input_shape, dtype="int32"),
      cond=conditioning,
      train=False,
  )
  state, params = flax.core.pop(variables, "params")
  del variables
  # weight_decay_mask = jax.tree_util.tree_map_with_path(
  #     lambda kp, x: (kp[-1].key != "bias"), params
  # )
  emb_traversal = flax.traverse_util.ModelParamTraversal(
      lambda p, _: "embedding_matrix" in p
  )

  all_false = jax.tree.map(lambda _: False, unbox_logicallypartioned(params))
  emb_mask = emb_traversal.update(lambda _: True, all_false)
  optimizer = optax.chain(
      optax.clip(config.clip) if config.clip > 0.0 else optax.identity(),
      optax.adamw(
          schedule_fn,
          b1=0.9,
          b2=config.b2,
          weight_decay=config.weight_decay,
          # mask=weight_decay_mask,
      ),
      optax.masked(optax.scale(config.emb_lr_multiplier), emb_mask),
  )

  return TrainState(
      step=0,
      rng=rng,
      params=params,
      ema_params=copy.deepcopy(params) if config.ema_rate > 0.0 else None,
      tx=optimizer,
      opt_state=optimizer.init(params),
      state=state,
    )


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_loss: metrics.Average.from_output("loss")
  eval_loss_diff: metrics.Average.from_output("loss_diff")
  eval_loss_prior: metrics.Average.from_output("loss_prior")
  eval_loss_recon: metrics.Average.from_output("loss_recon")


def create_train_metrics_class_from_keys(metric_keys):
  """Create train metrics collection from dictionary."""
  average_keys = [
      "loss",
      "loss_diff",
      "loss_prior",
      "loss_recon",
      "indices_probs_mse",
  ]
  stats = dict(
      (k, metrics.Average.from_output(k))
      if k in average_keys
      else (k, metrics.LastValue.from_output(k))
      for k in metric_keys
  )
  return metrics.Collection.create(**stats)


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


def loss_fn_diff(params, state, rng, model, batch, train=False, beta=1.0):
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
  else:
    raise ValueError("Unsupported targets/tasks.")

  if "label" in batch:
    conditioning = batch["label"].astype("int32")
  else:
    conditioning = None

  new_state = {}
  if train:
    (loss, loss_diff, loss_prior, loss_recon, model_stats), new_state = (
        model.apply(
            variables,
            x,
            cond=conditioning,
            train=train,
            rngs=rngs,
            beta=beta,
            mutable=list(state.keys()),
        )
    )
  else:
    loss, loss_diff, loss_prior, loss_recon, model_stats = model.apply(
        variables,
        x,
        cond=conditioning,
        train=train,
        rngs=rngs,
        beta=beta,
    )

  rescale_to_bpd = 1.0 / (jnp.prod(jnp.array(x.shape[1:])) * jnp.log(2.0))
  loss = loss * rescale_to_bpd
  metrics_dict = dict(
      loss_diff=loss_diff * rescale_to_bpd,
      loss_prior=loss_prior * rescale_to_bpd,
      loss_recon=loss_recon * rescale_to_bpd,
      **model_stats,
  )
  if train:
    return loss, (new_state, metrics_dict)
  return loss, metrics_dict


def loss_fn_ar(params, state, rng, model, batch, train=False):
  """Loss function."""
  rng, sample_rng = jax.random.split(rng)
  rngs = {"sample": sample_rng}
  if train:
    _, dropout_rng = jax.random.split(rng)
    rngs["dropout"] = dropout_rng

  variables = {"params": params, **state}
  if "image" in batch:
    raise NotImplementedError()
  elif "text" in batch:
    x = batch["text"][:, :-1]
    targets = batch["text"]
    conditioning = None
  else:
    raise ValueError("Unsupported targets/tasks.")

  new_state = {}
  if train:
    logits, new_state = model.apply(
        variables,
        x,
        cond=conditioning,
        train=train,
        rngs=rngs,
        mutable=list(state.keys()),
    )
  else:
    logits = model.apply(
        variables,
        x,
        cond=conditioning,
        train=train,
        rngs=rngs,
    )

  vocab_size = logits.shape[-1]
  logits = logits.reshape((-1, vocab_size))
  targets = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)

  loss = optax.softmax_cross_entropy(logits, targets).mean() / jnp.log(2.)
  metrics_dict = dict(
      loss_diff=0.,
      loss_prior=0.,
      loss_recon=0.,
  )

  if train:
    return loss, (new_state, metrics_dict)
  return loss, metrics_dict


@jax.jit
def merge_metrics(a_tree, b_tree):
  return jax.tree.map(lambda a, b: a + b, a_tree, b_tree)


def train_step(
    train_state: TrainState,
    batch: Mapping[str, jnp.ndarray],
    model: nn.Module,
    learning_rate_fn: Callable[[int], float],
    train_metrics_class: Any,
    beta: float = 1.0,
    ema_rate: float = 0.0,
    loss_fn_type: str = "diffusion",
    num_microbatches: int | None = None,
) -> tuple[TrainState, metrics.Collection]:
  """Perform a single training step."""
  logging.info("train_step(batch=%s)", batch)
  rng, new_rng = jax.random.split(train_state.rng)

  if loss_fn_type.endswith("diffusion"):
    loss_fn = functools.partial(loss_fn_diff, beta=beta)
  elif loss_fn_type == "ar":
    loss_fn = loss_fn_ar
  else:
    raise NotImplementedError()

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  if num_microbatches is None or num_microbatches <= 1:
    (loss, (new_state, metrics_dict)), grads = grad_fn(
        train_state.params, train_state.state, rng, model, batch, train=True,
    )
  else:
    batch_size = next(iter(batch.values())).shape[0]
    assert (
        batch_size % num_microbatches == 0
    ), "Batch size isn't divided evenly by num_microbatches."
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

      (loss, (new_state, metrics_dict)), grads = grad_fn(
          train_state.params, train_state_state, mbrng, model, mb, train=True,
      )
      return loss, metrics_dict, grads, new_state

    def per_microbatch_train_step(loop_cnt, carry):
      (rng, grad_accum, prev_metrics_dict, prev_loss, train_state_state) = (
          carry
      )
      loss, metrics_dict, grads, train_state_state = metrics_and_grad(
          loop_cnt, rng, train_state_state
      )

      grad_accum = jax.tree.map(jnp.add, grad_accum, grads)
      loss = jax.tree.map(jnp.add, loss, prev_loss)
      metrics_dict = jax.lax.cond(
          loop_cnt == 0,
          lambda _: metrics_dict,
          lambda _: merge_metrics(prev_metrics_dict, metrics_dict),
          None,
      )
      return rng, grad_accum, metrics_dict, loss, train_state_state

    # Initialize gradient accumulation loop state.
    accum_dtype = jnp.float32
    grad_accum_init = jax.tree.map(
        lambda x: jnp.zeros(x.shape, accum_dtype), train_state.params
    )
    loss_shape, initial_metrics_shape, _, _ = jax.eval_shape(
        metrics_and_grad,
        loop_cnt=0,
        rng=rng,
        train_state_state=train_state.state,
    )

    initial_metrics = {
        k: jnp.zeros(shape=v.shape, dtype=v.dtype)
        for k, v in initial_metrics_shape.items()
    }

    initial_loss = jnp.zeros(shape=loss_shape.shape, dtype=loss_shape.dtype)
    loop_init = (
        rng,
        grad_accum_init,
        initial_metrics,
        initial_loss,
        train_state.state,
    )
    _, grads, metrics_dict, loss, train_state_state = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init
    )
    loss = loss / num_microbatches
    metrics_dict = jax.tree.map(lambda x: x / num_microbatches, metrics_dict)
    new_state = train_state_state

  updates, new_opt_state = train_state.tx.update(
      grads, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  if ema_rate > 0.0:
    new_ema_params = jax.tree.map(
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

  new_metrics = train_metrics_class.single_from_model_output(
      loss=loss,
      learning_rate=learning_rate_fn(train_state.step),
      **metrics_dict,
  )
  return new_train_state, new_metrics


def eval_step(
    rng: jnp.ndarray,
    train_state: TrainState,
    batch: Mapping[str, jnp.ndarray],
    model: nn.Module,
    ema_rate: float = 0.0,
    loss_fn_type: str = "diffusion",
) -> metrics.Collection:
  """Compute the metrics for the given model in inference mode."""
  logging.info("eval_step(batch=%s)", batch)
  params = train_state.ema_params if ema_rate > 0.0 else train_state.params

  if loss_fn_type.endswith("diffusion"):
    loss_fn = loss_fn_diff
  elif loss_fn_type == "ar":
    loss_fn = loss_fn_ar
  else:
    raise NotImplementedError()

  loss, metrics_dict = loss_fn(
      params, train_state.state, rng, model, batch, train=False
  )
  return EvalMetrics.single_from_model_output(
      loss=loss,
      **metrics_dict,
  )


def reshape_batch(batch: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
  """Reshapes a batch to have the leading dimension for the local devices."""
  leading_dims = [jax.local_device_count(), -1]
  return jax.tree.map(
      lambda x: np.reshape(x, leading_dims + list(x.shape[1:])), batch
  )


def evaluate(
    p_eval_step: Any,
    rng: jnp.ndarray,
    train_state: TrainState,
    eval_loader: grain.DataLoader,
    mesh: Mesh,
    num_eval_steps: int = -1,
) -> EvalMetrics:
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = []
  eval_iter = multihost_dataloading.MultiHostDataLoadIterator(
      eval_loader, mesh
  )
  with utils.StepTraceContextHelper("eval", 0) as trace_context:
    # Use `iter` to reset the eval_loader before each evaluation.
    for step, batch in enumerate(eval_iter):
      rng, sub_rng = jax.random.split(rng)
      metrics_update = p_eval_step(sub_rng, train_state, batch)
      eval_metrics.append(metrics_update)
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break
      trace_context.next_step()
  if not eval_metrics:
    raise ValueError(f"Eval dataset {eval_loader} was empty.")
  eval_metrics = _process_metrics(eval_metrics, EvalMetrics)
  return eval_metrics


def _process_metrics(batch_metrics, matrics_class):
  batch_metrics = [jax.device_get(m) for m in batch_metrics]
  final_metrics = matrics_class.empty()
  for m in batch_metrics:
    final_metrics = final_metrics.merge(m)
  return final_metrics


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: epath.PathLike, olddir: epath.PathLike | None = None
) -> Mapping[str, Any]:
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    olddir: Optional directory to load old checkpoints from for partial loading.
      If provided, checkpoints will be loaded from olddir but saved to workdir.

  Returns:
    A dictionary that maps "train_state" to the TrainState and "train_iter" to
      the train iterator.
  """
  if config.use_hardware_rng:
    # https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#generating-random-numbers
    # https://jax.readthedocs.io/en/latest/jax.random.html#advanced-rng-configuration
    jax.config.update("jax_threefry_partitionable", True)
    rng = jax.random.PRNGKey(config.seed)
  else:
    rng = utils.get_rng(config.seed)
  logging.info("Using random seed %s.", rng)

  workdir = epath.Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)

  # Learning rate schedule.
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

  # Initialize model.
  model = model_utils.get_model(config)

  # Initialize train state.
  fsdp_parallel = (
      jax.device_count() // config.tensor_parallel
  )
  mesh = Mesh(
      mesh_utils.create_device_mesh(
          (fsdp_parallel, config.tensor_parallel)
      ),
      config.mesh_axis_names,
  )
  data_shape = input_pipeline.get_data_shape(config)
  init_state_partial = functools.partial(
      create_train_state,
      config=config,
      input_shape=(config.batch_size, *data_shape),
      schedule_fn=schedule_fn,
      model=model,
  )
  abstract_state = jax.eval_shape(init_state_partial, rng=rng)
  abstract_state_ub = unbox_logicallypartioned(abstract_state)
  parameter_overview.log_parameter_overview(
      abstract_state_ub.state,
      msg="############# state #############",
      include_stats=False,
  )
  parameter_overview.log_parameter_overview(
      abstract_state_ub.params,
      msg="############# params #############",
      include_stats=False,
  )

  state_logical_pspecs = nn.get_partition_spec(abstract_state)
  state_mesh_shardings = nn.logical_to_mesh_sharding(
      state_logical_pspecs, mesh, rules=config.logical_axis_rules
  )

  rng, train_state_rng = jax.random.split(rng)
  train_state = jax.jit(
      init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings
  )(rng=train_state_rng)
  train_state = unbox_logicallypartioned(train_state)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_manager = _get_checkpoint_manager(workdir)

  # Build input pipeline.
  rng, data_seed = jax.random.split(rng)
  data_seed = int(
      jax.random.randint(data_seed, [], minval=0, maxval=np.iinfo(np.int32).max)
  )
  # The input pipeline runs on each process and loads data for local TPUs.
  train_loader, eval_loaders, dataset_info = input_pipeline.create_datasets(config, data_seed)
  train_iter = multihost_dataloading.MultiHostDataLoadIterator(
      train_loader, mesh
  )

  # Retrieve data from previous checkpoints if possible.
  checkpointed_state = dict(
      train_state=train_state, train_iter=train_iter.local_iterator
  )

  restore_args = orbax_checkpoint.checkpoint_utils.construct_restore_args(
      abstract_state_ub,
      state_mesh_shardings,
  )

  restore_kwargs = {"train_state": {"restore_args": restore_args}}
  if checkpoint_manager.latest_step() is not None:
    checkpointed_state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), items=checkpointed_state,
        restore_kwargs=restore_kwargs,
    )

  train_state = checkpointed_state["train_state"]
  train_iter.local_iterator = checkpointed_state["train_iter"]

  # Start training.
  train_metrics_class = create_train_metrics_class(config)
  train_step_func = functools.partial(
      train_step,
      model=model,
      learning_rate_fn=schedule_fn,
      train_metrics_class=train_metrics_class,
      beta=config.beta,
      ema_rate=config.ema_rate,
      loss_fn_type=config.model_type,
      num_microbatches=config.get("num_microbatches", None),
  )
  if config.check_nans:
    train_step_func = checkify.checkify(
        train_step_func, errors=checkify.float_checks
    )

  data_sharding = NamedSharding(mesh, P(config.data_sharding))
  p_train_step = jax.jit(
      train_step_func,
      in_shardings=(state_mesh_shardings, data_sharding),
      out_shardings=(state_mesh_shardings, None),
      donate_argnames=["train_state"],
  )
  p_eval_step = jax.jit(
      functools.partial(
          eval_step,
          model=model,
          ema_rate=config.ema_rate,
          loss_fn_type=config.model_type,
      ),
      in_shardings=(None, state_mesh_shardings, data_sharding),
      out_shardings=(None),
  )

  p_generate = jax.jit(
      functools.partial(
          sampling.simple_generate,
          batch_size=4,
          model=model,
          conditioning=None,
      ),
      in_shardings=(None, state_mesh_shardings),
      out_shardings=(None),
  )

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
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
  initial_step = int(train_state.step)
  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
    for step in range(initial_step + 1, num_train_steps + 1):
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = next(train_iter)

        if config.check_nans:
          errs, (train_state, metrics_update) = p_train_step(train_state, batch)
          errs.throw()
        else:
          train_state, metrics_update = p_train_step(train_state, batch)
        train_metrics.append(metrics_update)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        # train metrics are averaged over log_loss_every_steps.
        train_metrics = _process_metrics(train_metrics, train_metrics_class)
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = []

      if step == 1 or step % config.eval_every_steps == 0 or is_last_step:
        for split, eval_loader in eval_loaders.items():
          rng, eval_rng = jax.random.split(rng)
          with report_progress.timed("eval"):
            eval_metrics = evaluate(
                p_eval_step,
                eval_rng,
                train_state,
                eval_loader,
                mesh,
                config.num_eval_steps,
            )
          eval_metrics_cpu = jax.tree.map(np.array, eval_metrics.compute())
          eval_metrics_cpu = {
              split + k[4:]: v for k, v in eval_metrics_cpu.items()
          }
          writer.write_scalars(step, eval_metrics_cpu)

        if jax.process_index() == 0:
          if config.model_type == "gaussian_diffusion":
            params = train_state.params
            state = train_state.state
            embeddings = model.apply(
                {"params": params, **state}, method=model.embeddings
            )
            if config.dataset == "text8":
              annotations = dict((i, chr(i + 97)) for i in range(26))
              annotations.update({26: " "})
            else:
              annotations = None
              utils.plot_embeddings(step, workdir, embeddings, annotations)

        with report_progress.timed("sample"):
          _, sample_rng = jax.random.split(rng)
          if config.model_type.endswith("diffusion"):
            samples = p_generate(sample_rng, train_state)
          else:
            logging.info("Haven't implemented sharded sampling for AR.")

          if config.task_type == "image"and config.model_type.endswith(
              "diffusion"
          ):
            sample_grid = utils.generate_image_grids(samples)
            writer.write_images(step, {"samples": sample_grid})
          elif config.task_type == "text" and config.model_type.endswith(
              "diffusion"
          ):
            tokenizer = dataset_info["tokenizer"]
            texts = utils.detokenize_texts(samples, tokenizer)
            writer.write_texts(step, {"samples": texts})

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          checkpoint_manager.save(
              step,
              items=dict(
                  train_state=train_state,
                  train_iter=train_iter.local_iterator,
              ),
          )

  logging.info("Finishing training at step %d", num_train_steps)
  return checkpointed_state
