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

"""Utilities for partial loading of model checkpoints."""

import copy
import logging
from typing import Any, Tuple

import jax
from clu import parameter_overview
from orbax import checkpoint as orbax_checkpoint
from flax.training import orbax_utils


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

        new_params = copy.deepcopy(train_state.params)
        new_ema_params = (
            copy.deepcopy(train_state.ema_params)
            if hasattr(train_state, "ema_params") and train_state.ema_params is not None
            else None
        )

        # Copy all loaded parameters that exist in both models
        params_copied = []
        params_skipped = []

        for key in loaded_params:
            if key in new_params:
                new_params[key] = copy.deepcopy(loaded_params[key])
                params_copied.append(key)
            else:
                params_skipped.append(key)

        # Copy EMA parameters if they exist
        if new_ema_params is not None and loaded_ema_params is not None:
            for key in loaded_ema_params:
                if key in new_ema_params:
                    new_ema_params[key] = copy.deepcopy(loaded_ema_params[key])

        logging.info(f"Copied parameters: {params_copied}")
        if params_skipped:
            logging.info(f"Skipped parameters (not in new model): {params_skipped}")

        # Find parameters that are new in the current model
        new_params_only = [key for key in new_params if key not in loaded_params]
        if new_params_only:
            logging.info(
                f"New parameters (kept from fresh initialization): {new_params_only}"
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
    train_state, train_iter, checkpoint_manager: orbax_checkpoint.CheckpointManager
) -> Tuple[Any, Any]:
    """Perform standard checkpoint loading.

    Args:
        train_state: Train state to load into
        train_iter: Training iterator
        checkpoint_manager: Orbax checkpoint manager

    Returns:
        Tuple of (loaded_train_state, loaded_train_iter)
    """
    checkpointed_state = {"train_state": train_state}
    if train_iter is not None:
        checkpointed_state["train_iter"] = train_iter

    restore_args = orbax_utils.restore_args_from_target(checkpointed_state)
    checkpointed_state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), items=checkpointed_state, restore_kwargs={'restore_args': restore_args}
    )

    loaded_train_state = checkpointed_state["train_state"]
    loaded_train_iter = (
        checkpointed_state["train_iter"]
        if "train_iter" in checkpointed_state
        else train_iter
    )

    return loaded_train_state, loaded_train_iter
