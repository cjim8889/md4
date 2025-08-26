import copy
import logging
from typing import Any, Tuple

import jax
import jax.tree_util as tree_util
import ml_collections
from clu import parameter_overview
from etils import epath
from orbax import checkpoint as orbax_checkpoint


def get_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    create: bool = True,
) -> orbax_checkpoint.CheckpointManager:
    """Create a checkpoint manager with preemption tolerance."""
    # Use checkpoint_dir from config if specified, otherwise default to workdir/checkpoints
    if (
        hasattr(config, "checkpoint_dir")
        and config.checkpoint_dir
        and isinstance(config.checkpoint_dir, str)
    ):
        checkpoint_dir = epath.Path(config.checkpoint_dir)
    else:
        checkpoint_dir = epath.Path(workdir) / "checkpoints"

    # Ensure checkpoint_every_steps is an integer
    checkpoint_every_steps = config.get("checkpoint_every_steps", 10000)
    if not isinstance(checkpoint_every_steps, int):
        checkpoint_every_steps = 10000

    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        options=orbax_checkpoint.CheckpointManagerOptions(
            create=create,
            best_fn=lambda x: x.get("validation_loss", x.get("loss", float("inf"))),
            best_mode="min",
            max_to_keep=20,
            save_interval_steps=checkpoint_every_steps,
        ),
    )


def get_checkpoint_managers(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
) -> tuple[orbax_checkpoint.CheckpointManager, orbax_checkpoint.CheckpointManager]:
    """Get save and load checkpoint managers with preemption tolerance.

    Args:
        config: Configuration to use.
        workdir: Working directory for saving checkpoints.
        olddir: Optional directory to load old checkpoints from. If provided,
            load manager will point to olddir, otherwise same as save manager.

    Returns:
        A tuple of (save_manager, load_manager).
    """
    # Create checkpoint manager for saving (new checkpoints)
    save_checkpoint_manager = get_checkpoint_manager(config, workdir, create=True)

    # Create checkpoint manager for loading (old checkpoints if olddir is provided)
    if olddir is not None:
        load_checkpoint_manager = get_checkpoint_manager(config, olddir, create=False)
    else:
        load_checkpoint_manager = save_checkpoint_manager

    return save_checkpoint_manager, load_checkpoint_manager


# Partial loading utilities


def should_use_partial_loading(config) -> bool:
    """Check if the config should use partial loading.

    Args:
        config: Configuration object

    Returns:
        True if partial loading should be used, False otherwise
    """
    return (
        hasattr(config, "old_config")
        and config.old_config is not None
        and config.get("partial_load", False)
    )


def get_old_config(config):
    """Get the old molecular configuration.

    Args:
        config: Current configuration that contains old_config path

    Returns:
        Configuration object for the old molecular model
    """
    import importlib.util
    import sys
    from pathlib import Path

    if not hasattr(config, "old_config") or config.old_config is None:
        raise ValueError(
            "Config must have 'old_config' attribute specifying the old config path"
        )

    old_config_path = config.old_config
    logging.info(f"Loading old config from: {old_config_path}")

    # Handle different path formats
    if old_config_path.startswith("md4/"):
        # Relative path from md4 package - convert to module path
        # e.g., "md4/configs/md4/molecular.py" -> "md4.configs.md4.molecular"
        module_path = old_config_path.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]  # Remove .py extension

        # Import as module
        try:
            module = importlib.import_module(module_path)
            return module.get_config()
        except ImportError as e:
            logging.error(f"Failed to import {module_path}: {e}")
            raise ImportError(
                f"Could not import old config module {module_path}"
            ) from e
    else:
        # Absolute file path
        config_file = Path(old_config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Old config file not found: {old_config_path}")

        # Load module from file path
        spec = importlib.util.spec_from_file_location("old_config", config_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from {old_config_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["old_config"] = module
        spec.loader.exec_module(module)
        return module.get_config()


def format_parameter_path(path):
    """Format a parameter path tuple for readable logging.

    Args:
        path: Tuple representing the parameter path from tree_flatten_with_path

    Returns:
        String representation of the path
    """
    path_parts = []
    for part in path:
        if hasattr(part, "key"):  # DictKey
            path_parts.append(f".{part.key}")
        elif hasattr(part, "idx"):  # SequenceKey
            path_parts.append(f"[{part.idx}]")
        else:
            path_parts.append(f".{part}")
    return "".join(path_parts).lstrip(".")


def copy_matching_parameters(old_params, new_params):
    """Copy parameters from old tree to new tree using tree flattening.

    This function flattens both parameter trees and copies values from the old tree
    to the new tree for all matching parameter paths, leaving non-matching parameters
    in the new tree unchanged.

    Args:
        old_params: Parameter tree from the old model
        new_params: Parameter tree from the new model (will be modified)

    Returns:
        Tuple of (updated_new_params, copied_keys, skipped_keys, new_only_keys)
    """
    # Flatten both parameter trees to get all parameter paths
    old_flat, old_tree_def = tree_util.tree_flatten_with_path(old_params)
    new_flat, new_tree_def = tree_util.tree_flatten_with_path(new_params)

    # Create dictionaries mapping parameter paths to values
    old_param_dict = {path: value for path, value in old_flat}
    new_param_dict = {path: value for path, value in new_flat}

    # Track which parameters we copy, skip, or are new
    copied_keys = []
    skipped_keys = []
    new_only_keys = []

    # Create updated parameter dictionary
    updated_param_dict = {}

    # Process all parameters in the new model
    for path in new_param_dict:
        if path in old_param_dict:
            # Copy from old model if the path exists
            updated_param_dict[path] = copy.deepcopy(old_param_dict[path])
            copied_keys.append(path)
        else:
            # Keep new initialization if path doesn't exist in old model
            updated_param_dict[path] = new_param_dict[path]
            new_only_keys.append(path)

    # Find parameters that exist in old model but not in new model
    for path in old_param_dict:
        if path not in new_param_dict:
            skipped_keys.append(path)

    # Reconstruct the parameter tree with updated values
    updated_values = [updated_param_dict[path] for path, _ in new_flat]
    updated_params = tree_util.tree_unflatten(new_tree_def, updated_values)

    return updated_params, copied_keys, skipped_keys, new_only_keys


def partial_load_checkpoint(
    config,
    train_state,
    train_iter,
    checkpoint_manager: orbax_checkpoint.CheckpointManager,
    create_train_state_fn,
    schedule_fn,
    per_device_batch_size: int,
    data_shape: Tuple[int, ...],
) -> Tuple[Any, Any]:
    """Perform partial checkpoint loading for molecular finetuning.

    This function loads a checkpoint from an old model configuration and copies
    compatible parameters to a new model configuration. Parameters that don't
    exist in the old model (like newly added layers) are kept from the fresh
    initialization.

    Args:
        config: New model configuration
        train_state: Fresh train state for the new model
        train_iter: Training iterator
        checkpoint_manager: Orbax checkpoint manager
        create_train_state_fn: Function to create train state
        schedule_fn: Learning rate schedule function
        per_device_batch_size: Batch size per device
        data_shape: Shape of input data

    Returns:
        Tuple of (updated_train_state, updated_train_iter)
    """
    logging.info("Performing partial checkpoint loading for molecular_finetune config")

    try:
        # Get the old config
        old_config = get_old_config(config)

        # Create old model and train state for loading the checkpoint
        old_rng = jax.random.PRNGKey(88)  # Use a fixed seed for reproducibility
        old_model, old_optimizer, old_train_state, _ = create_train_state_fn(
            old_config,
            old_rng,
            input_shape=(per_device_batch_size,) + data_shape,
            schedule_fn=schedule_fn,
        )

        # Prepare checkpoint state for old model
        old_checkpointed_state = {"train_state": old_train_state}
        if config.dataset not in ["pubchem_large", "msg_finetune"]:
            old_checkpointed_state["train_iter"] = train_iter

        # Load checkpoint into old train state
        old_checkpointed_state = checkpoint_manager.restore(
            config.get("old_checkpoint_steps", checkpoint_manager.latest_step()),
            items=old_checkpointed_state,
        )

        loaded_train_state = old_checkpointed_state["train_state"]
        loaded_params = loaded_train_state.params
        loaded_ema_params = getattr(loaded_train_state, "ema_params", None)

        logging.info(f"Loaded checkpoint at step: {loaded_train_state.step}")

        # Use tree-based parameter copying for better matching
        logging.info("Copying main parameters using tree flattening...")
        new_params, copied_keys, skipped_keys, new_only_keys = copy_matching_parameters(
            loaded_params, train_state.params
        )

        # Log parameter copying results
        logging.info(f"Copied {len(copied_keys)} parameter paths")
        if len(copied_keys) <= 20:  # Only log individual paths if not too many
            for key in copied_keys[:10]:  # Show first 10
                logging.info(f"  Copied: {format_parameter_path(key)}")
            if len(copied_keys) > 10:
                logging.info(f"  ... and {len(copied_keys) - 10} more")

        if skipped_keys:
            logging.info(f"Skipped {len(skipped_keys)} parameters (not in new model)")
            if len(skipped_keys) <= 10:
                for key in skipped_keys:
                    logging.info(f"  Skipped: {format_parameter_path(key)}")

        if new_only_keys:
            logging.info(
                f"Kept {len(new_only_keys)} new parameters from fresh initialization"
            )
            if len(new_only_keys) <= 10:
                for key in new_only_keys:
                    logging.info(f"  New: {format_parameter_path(key)}")

        # Copy EMA parameters if they exist using the same tree-based approach
        new_ema_params = None
        if hasattr(train_state, "ema_params") and train_state.ema_params is not None:
            if loaded_ema_params is not None:
                logging.info("Copying EMA parameters using tree flattening...")
                new_ema_params, ema_copied, ema_skipped, ema_new = (
                    copy_matching_parameters(loaded_ema_params, train_state.ema_params)
                )
                logging.info(
                    f"EMA parameters: copied {len(ema_copied)}, skipped {len(ema_skipped)}, new {len(ema_new)}"
                )
            else:
                # Keep fresh EMA parameters if old model doesn't have them
                new_ema_params = train_state.ema_params
                logging.info(
                    "Old model has no EMA parameters, keeping fresh initialization"
                )

        # Update the new train state with copied parameters
        updated_train_state = train_state.replace(
            params=new_params,
            ema_params=new_ema_params,
        )

        logging.info("Created updated train state.")
        parameter_overview.log_parameter_overview(
            updated_train_state.params,
            msg="Updated Train State Parameters Overview",
        )

        # Handle train_iter if it exists in the old checkpoint
        updated_train_iter = train_iter
        if "train_iter" in old_checkpointed_state:
            updated_train_iter = old_checkpointed_state["train_iter"]

        logging.info("Successfully completed partial checkpoint loading")
        return updated_train_state, updated_train_iter

    except Exception as e:
        logging.error(f"Error during partial loading: {e}")
        logging.info("Falling back to standard checkpoint loading")
        raise e


def standard_checkpoint_loading(
    train_state,
    checkpoint_manager: orbax_checkpoint.CheckpointManager,
) -> Any:
    """Perform standard checkpoint loading using new Orbax API.

    Args:
        train_state: Train state to load into
        checkpoint_manager: Orbax checkpoint manager

    Returns:
        Loaded train state
    """
    logging.info("Performing standard checkpoint loading...")

    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        logging.info("No checkpoint found, using fresh train state")
        return train_state

    logging.info(f"Loading checkpoint from step: {latest_step}")

    # Load checkpoint with StandardRestore
    loaded_train_state = checkpoint_manager.restore(
        latest_step, args=orbax_checkpoint.args.StandardRestore(train_state)
    )
    step = getattr(loaded_train_state, "step", "unknown")
    logging.info(f"Successfully loaded checkpoint at step: {step}")
    return loaded_train_state
