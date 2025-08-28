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

"""Deterministic input pipeline."""

from typing import Any

import jax
from ml_collections import config_dict

from md4 import (
    input_pipeline_msg_finetune,
    input_pipeline_pubchem_large,
    input_pipeline_pubchem_large_text,
)


def get_data_shape(config):
    return config.data_shape


def get_num_train_steps(config: config_dict.ConfigDict) -> int:
    """Calculates the total number of training steps."""
    if config.num_train_steps > 0:
        return config.num_train_steps
    raise NotImplementedError()


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Create data loaders for training and evaluation.

    Args:
      config: Configuration to use.
      seed: Seed for shuffle and random operations in the training dataset.

    Returns:
      A tuple with the training dataset loader, the evaluation dataset
      loader, and a dictionary of other infos.
    """
    info = {}
    assert config.batch_size % jax.process_count() == 0

    if config.dataset == "pubchem_large":
        train_dataset, eval_dataset, pubchem_info = (
            input_pipeline_pubchem_large.create_pubchem_datasets(config, seed)
        )
        info.update(pubchem_info)
        return train_dataset, eval_dataset, info
    elif config.dataset == "pubchem_large_text":
        train_dataset, eval_dataset, pubchem_info = (
            input_pipeline_pubchem_large_text.create_pubchem_datasets(config, seed)
        )
        info.update(pubchem_info)
        return train_dataset, eval_dataset, info
    elif config.dataset == "msg_finetune":
        train_dataset, eval_dataset, msg_info = (
            input_pipeline_msg_finetune.create_msg_finetune_datasets(config, seed)
        )
        info.update(msg_info)
        return train_dataset, eval_dataset, info
    else:
        raise NotImplementedError(
            "Only pubchem_large, pubchem_large_text, and msg_finetune datasets are supported."
        )
