"""Utilities for creating and managing training state."""

import copy
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import flax.core
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import ml_collections
import optax
from absl import logging
from clu import metrics, parameter_overview
from flax import traverse_util
from jax.experimental import multihost_utils

from md4.models import utils as model_utils


def get_default_logical_axis_rules():
    """Get default logical axis rules for model sharding.

    Returns:
        List of tuples mapping logical axis names to physical mesh axis names.
    """
    return [
        ("batch", "data"),
        ("hidden", "model"),
        ("attn_qkv", "model"),
        ("attn_o", "model"),
        ("ff_mlp", "model"),
        ("embed_vocab", "model"),
        ("input_embed", "model"),
        ("cross_attn", "model"),
        ("cond", "model"),
        ("cond_input", "model"),
        ("cond_hidden", "model"),
        ("cond_output", "model"),
        ("vocab", "model"),
        # leave sequence/time unsharded
    ]


@flax.struct.dataclass
class TrainState:
    """State of the model and the training.

    This includes parameters, statistics and optimizer.
    """

    rng: jnp.ndarray
    step: int
    params: Any
    ema_params: Any
    opt_state: optax.OptState
    state: Any


def get_conditioning_from_batch(batch, dtype=jnp.float32):
    """Extract conditioning information from batch data.

    Handles all combinations of fingerprint, true_fingerprint, and atom_types.

    Args:
        batch: Batch data dictionary
        dtype: Data type to use for floating point conditioning data
    """
    if "label" in batch:
        return batch["label"].astype("int32")

    # Build conditioning dict based on available fields
    conditioning = {}

    # Handle fingerprint (prioritize true_fingerprint if available)
    if "true_fingerprint" in batch:
        conditioning["true_fingerprint"] = batch["true_fingerprint"].astype(dtype)

    if "fingerprint" in batch:
        conditioning["cross_conditioning"] = batch["fingerprint"].astype(dtype)

    # Handle atom_types
    if "atom_types" in batch:
        conditioning["atom_types"] = batch["atom_types"].astype("int32")

    # Return None if no conditioning information is available
    return conditioning if conditioning else None


def get_dummy_conditioning(config, input_shape, dtype=jnp.float32):
    """Create dummy conditioning for model initialization.

    Handles all combinations based on config settings.

    Args:
        config: Configuration dict
        input_shape: Input shape tuple
        dtype: Data type to use for floating point conditioning data
    """
    if config.classes > 0:
        return jnp.zeros(input_shape[0], dtype="int32")

    conditioning = {}

    # Handle fingerprint dimension
    if config.fingerprint_dim > 0:
        fingerprint_dim = (
            config.raw_fingerprint_dim
            if config.get("raw_fingerprint_dim", 0) > 0
            else config.fingerprint_dim
        )
        conditioning["cross_conditioning"] = jnp.zeros(
            (input_shape[0], fingerprint_dim), dtype=dtype
        )

    if (
        config.get("raw_fingerprint_dim", 0) > 0
        and config.raw_fingerprint_dim != config.fingerprint_dim
    ):
        conditioning["true_fingerprint"] = jnp.zeros(
            (input_shape[0], config.fingerprint_dim), dtype=dtype
        )

    # Return None if no conditioning information is available
    return conditioning if conditioning else None


def create_metrics_class_from_keys(metric_keys):
    """Create train/eval metrics collection from dictionary."""
    average_keys = []
    stats = dict(
        (k, metrics.Average.from_output(k))
        if (k in average_keys) or ("loss" in k)
        else (k, metrics.LastValue.from_output(k))
        for k in metric_keys
    )
    return metrics.Collection.create(**stats)


def _should_freeze_parameter(path, v, config: ml_collections.ConfigDict) -> bool:
    """Determine if a parameter should be frozen based on config.

    Args:
        path: Parameter path tuple
        v: Parameter value
        config: Configuration with frozen_paths and unfrozen_paths

    Returns:
        True if parameter should be frozen, False otherwise
    """
    # Check unfrozen paths first (these take precedence)
    unfrozen_paths = config.get("unfrozen_paths", [])
    if isinstance(unfrozen_paths, (list, tuple)):
        for unfrozen_path in unfrozen_paths:
            if unfrozen_path in path:
                return False

    # If frozen_paths is specified as a non-empty list, only freeze those paths
    frozen_paths = config.get("frozen_paths", [])
    if isinstance(frozen_paths, (list, tuple)):
        if len(frozen_paths) > 0:
            # Non-empty frozen_paths: only freeze specified paths
            for frozen_path in frozen_paths:
                if frozen_path in path:
                    return True
            return False  # Not in frozen_paths, so don't freeze
        else:
            # Empty frozen_paths: freeze all except unfrozen_paths
            return True

    # Default behavior: freeze all except unfrozen_paths
    return True


def _should_initialize_adapter(path, config: ml_collections.ConfigDict) -> bool:
    """Determine if a parameter should be adapter-initialized based on config.

    Args:
        path: Parameter path tuple
        config: Configuration with adapter_init_paths

    Returns:
        True if parameter should be adapter-initialized, False otherwise
    """
    adapter_init_paths = config.get("adapter_init_paths", [])
    if adapter_init_paths and isinstance(adapter_init_paths, (list, tuple)):
        for adapter_path in adapter_init_paths:
            if adapter_path in path:
                return True
    return False


def _initialize_adapter_weights(
    path, v: jnp.ndarray, config: ml_collections.ConfigDict
) -> jnp.ndarray:
    """Initialize adapter weights with special initialization.

    Args:
        path: Parameter path tuple
        v: Parameter value
        config: Configuration

    Returns:
        Initialized parameter value
    """
    if not _should_initialize_adapter(path, config):
        return v

    # Check if this is a kernel or bias parameter by looking at the path
    if "kernel" in path:
        return jnp.eye(v.shape[0], v.shape[1], dtype=v.dtype)
    elif "bias" in path:
        return jnp.zeros(v.shape, dtype=v.dtype)

    return v


def create_train_metrics_class_from_keys(metric_keys):
    """Create train metrics collection from dictionary."""
    average_keys = [
        "loss",
        "loss_diff",
        "loss_prior",
        "loss_recon",
    ]
    stats = dict(
        (k, metrics.Average.from_output(k))
        if k in average_keys
        else (k, metrics.LastValue.from_output(k))
        for k in metric_keys
    )
    return metrics.Collection.create(**stats)


def create_train_metrics_class():
    metric_keys = sorted(
        ["loss_prior", "loss_diff", "loss_recon", "loss", "learning_rate"]
    )

    logging.info("metric_keys: %s", metric_keys)
    return create_train_metrics_class_from_keys(metric_keys)


def show_pspec(name, x):
    try:
        print(name, getattr(x, "sharding", None))
    except:
        pass


def create_sharded_train_state(
    config: ml_collections.ConfigDict,
    x_sharding: jax.sharding.NamedSharding,
    mesh: jax.sharding.Mesh,
    rng: jnp.ndarray,
    input_shape: Sequence[int] | Mapping[str, Sequence[int]],
    schedule_fn: Callable[[Any], Any],
):
    model = model_utils.get_model(config)

    conditioning = get_dummy_conditioning(
        config, input_shape, dtype=getattr(config, "dtype", jnp.float32)
    )
    dummy_input = jnp.ones(input_shape, dtype="int32")

    def _init_fn(key, x, conditioning, model: nn.Module, optimizer):
        rng, sample_rng, init_rng = jax.random.split(key, 3)
        variables = model.init(
            {"sample": sample_rng, "params": init_rng},
            x,
            cond=conditioning,
            train=False,
        )
        internal_state, params = flax.core.pop(variables, "params")
        del variables

        state = TrainState(
            step=0,
            rng=rng,
            params=params,
            ema_params=copy.deepcopy(params) if config.ema_rate > 0.0 else None,
            opt_state=optimizer.init(params),
            state=internal_state,
        )

        return state

    if config.get("scale_by_muon", False):
        adam = optax.contrib.muon(
            schedule_fn,
            adam_b1=0.9,
            adam_b2=config.b2,
            weight_decay=config.weight_decay,
        )
    else:
        adam = optax.adamw(
            schedule_fn,
            b1=0.9,
            b2=config.b2,
            weight_decay=config.weight_decay,
        )



    chains = [
        optax.clip(config.clip) if config.clip > 0.0 else optax.identity(),
        adam,
    ]

    optimizer = optax.chain(
        *chains,
        # optax.zero_nans(), # This is more tricky to use when fsdp is enabled
    )

    # Get logical axis rules from config, with fallback to defaults
    logical_axis_rules = getattr(
        config, "logical_axis_rules", get_default_logical_axis_rules()
    )

    # ---- trace shapes and derive sharding under mesh + rules ----
    with mesh, nn.logical_axis_rules(logical_axis_rules):
        logical_abstract_state = jax.eval_shape(
            functools.partial(_init_fn, model=model, optimizer=optimizer),
            rng,
            dummy_input,
            conditioning,
        )
        logical_state_pspec = nn.get_partition_spec(logical_abstract_state)
        state_sharding = nn.logical_to_mesh_sharding(
            logical_state_pspec, mesh=mesh, rules=logical_axis_rules
        )

        jitted_init_fn = jax.jit(
            _init_fn,
            static_argnums=(3, 4),
            in_shardings=(None, x_sharding, x_sharding),
            out_shardings=state_sharding,
        )
        initialized_state = jitted_init_fn(
            rng, dummy_input, conditioning, model, optimizer
        )

    initialized_state = nn.meta.unbox(initialized_state)
    parameter_overview.log_parameter_overview(
        initialized_state.state,
        msg="############# state #############",
        jax_logging_process=0,
    )

    gathered_params = multihost_utils.process_allgather(initialized_state.params)
    parameter_overview.log_parameter_overview(
        gathered_params, msg="############# params #############", jax_logging_process=0
    )

    metrics_class = create_train_metrics_class()

    return (model, optimizer, initialized_state, metrics_class, state_sharding)


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,
    input_shape: Sequence[int] | Mapping[str, Sequence[int]],
    schedule_fn: Callable[[Any], Any],
) -> tuple[Any, optax.GradientTransformation, TrainState, metrics.Collection]:
    """Create and initialize the model with optional parameter freezing.

    This function replaces both create_train_state and create_frozen_train_state
    from the original train.py file, providing unified functionality based on
    the config settings.

    Args:
        config: Configuration dict with optional frozen/unfrozen paths
        rng: Random number generator
        input_shape: Input shape for model initialization
        schedule_fn: Learning rate schedule function

    Returns:
        Tuple of (model, optimizer, train_state, metrics_class)
    """
    model = model_utils.get_model(config)

    conditioning = get_dummy_conditioning(
        config, input_shape, dtype=getattr(config, "dtype", jnp.float32)
    )
    rng, sample_rng, init_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones(input_shape, dtype="int32")

    output, variables = model.init_with_output(
        {"sample": sample_rng, "params": init_rng},
        dummy_input,
        cond=conditioning,
        train=False,
    )

    metric_keys = sorted(list(output.keys()) + ["learning_rate"])
    logging.info("metric_keys: %s", metric_keys)
    metrics_class = create_metrics_class_from_keys(metric_keys)
    state, params = flax.core.pop(variables, "params")
    del variables
    parameter_overview.log_parameter_overview(
        state, msg="############# state #############"
    )
    parameter_overview.log_parameter_overview(
        jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.LogicallyPartitioned) else x,
            params,
            is_leaf=lambda k: isinstance(k, nn.LogicallyPartitioned),
        ),
        msg="############# params #############",
    )

    # Initialize adapter weights if specified
    if config.get("adapter_init_paths"):
        adapter_init_fn = functools.partial(_initialize_adapter_weights, config=config)
        params = traverse_util.path_aware_map(adapter_init_fn, params)

    # Create optimizer based on whether we're using parameter freezing
    use_freezing = config.get("frozen", False) and (
        config.get("frozen_paths") or config.get("unfrozen_paths")
    )

    if use_freezing:
        # Create freeze mask for selective training
        freeze_fn = functools.partial(_should_freeze_parameter, config=config)
        mask = traverse_util.path_aware_map(freeze_fn, params)

        adam = optax.transforms.selective_transform(
            optax.adamw(
                schedule_fn,
                b1=0.9,
                b2=config.b2,
                weight_decay=config.weight_decay,
            ),
            freeze_mask=mask,
        )
    else:
        # Standard optimizer without freezing
        adam = optax.adamw(
            schedule_fn,
            b1=0.9,
            b2=config.b2,
            weight_decay=config.weight_decay,
        )

    optimizer = optax.chain(
        optax.clip(config.clip) if config.clip > 0.0 else optax.identity(),
        adam,
        optax.zero_nans(),
    )

    return (
        model,
        optimizer,
        TrainState(
            step=0,
            rng=rng,
            params=params,
            ema_params=copy.deepcopy(params) if config.ema_rate > 0.0 else None,
            opt_state=optimizer.init(params),
            state=state,
        ),
        metrics_class,
    )
