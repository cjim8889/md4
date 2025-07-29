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

from typing import Any, Tuple
import logging
import jax
from orbax import checkpoint as orbax_checkpoint
from md4.configs.md4 import molecular
import ml_collections
from collections.abc import Callable, Mapping, Sequence


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
        and "molecular.py" in str(config.old_config)
        and config.get("partial_load", False)
    )


def get_old_config():
    """Get the old molecular configuration.

    Returns:
        Configuration object for the old molecular model
    """
    return molecular.get_config()


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
        old_config = get_old_config()

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
            checkpoint_manager.latest_step(),
            items=old_checkpointed_state,
        )

        loaded_train_state = old_checkpointed_state["train_state"]
        loaded_params = loaded_train_state.params
        loaded_ema_params = getattr(loaded_train_state, "ema_params", None)

        logging.info(f"Loaded checkpoint at step: {loaded_train_state.step}")

        # Copy compatible parameters to the new train state
        new_params = dict(train_state.params)
        new_ema_params = (
            dict(train_state.ema_params)
            if hasattr(train_state, "ema_params") and train_state.ema_params is not None
            else None
        )

        # Copy all loaded parameters that exist in both models
        params_copied = []
        params_skipped = []

        for key in loaded_params:
            if key in new_params:
                new_params[key] = loaded_params[key]
                params_copied.append(key)
            else:
                params_skipped.append(key)

        # Copy EMA parameters if they exist
        if new_ema_params is not None and loaded_ema_params is not None:
            for key in loaded_ema_params:
                if key in new_ema_params:
                    new_ema_params[key] = loaded_ema_params[key]

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

    checkpointed_state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), items=checkpointed_state
    )

    loaded_train_state = checkpointed_state["train_state"]
    loaded_train_iter = (
        checkpointed_state["train_iter"]
        if "train_iter" in checkpointed_state
        else train_iter
    )

    return loaded_train_state, loaded_train_iter
