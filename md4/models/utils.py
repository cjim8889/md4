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

"""Model utils."""

import ml_collections

from md4.models.diffusion import genmd4
from md4.models.diffusion import md4


def get_model(config: ml_collections.ConfigDict):
  """Get model instances."""
  if config.model_type == "md4":
    return md4.MD4(
        config.data_shape,
        cont_time=config.cont_time,
        timesteps=config.timesteps,
        feature_dim=config.feature_dim,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_dit_layers=config.n_dit_layers,
        dit_num_heads=config.dit_num_heads,
        dit_hidden_size=config.dit_hidden_size,
        ch_mult=config.ch_mult,
        vocab_size=config.vocab_size,
        noise_schedule_type=config.noise_schedule,
        dropout_rate=config.dropout_rate,
        use_attn_dropout=config.get("use_attn_dropout", True),
        mlp_type=config.mlp_type,
        depth_scaled_init=config.depth_scaled_init,
        cond_type=config.cond_type,
        outside_embed=config.outside_embed,
        time_features=config.time_features,
        classes=config.classes,
        sampler=config.sampler,
        sampling_grid=config.sampling_grid,
        topp=config.topp,
        model_sharding=config.get("model_sharding", False),
        fingerprint_dim=config.get("fingerprint_dim", 0),
        raw_fingerprint_dim=config.get("raw_fingerprint_dim", 0),
        atom_type_size=config.get("atom_type_size", 0),
    )
  elif config.model_type == "genmd4":
    return genmd4.GenMD4(
        config.data_shape,
        cont_time=config.cont_time,
        timesteps=config.timesteps,
        feature_dim=config.feature_dim,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_dit_layers=config.n_dit_layers,
        dit_num_heads=config.dit_num_heads,
        dit_hidden_size=config.dit_hidden_size,
        ch_mult=config.ch_mult,
        vocab_size=config.vocab_size,
        noise_schedule_type=config.noise_schedule,
        power_init=config.power_init,
        dropout_rate=config.dropout_rate,
        use_attn_dropout=config.get("use_attn_dropout", True),
        mlp_type=config.mlp_type,
        depth_scaled_init=config.depth_scaled_init,
        cond_type=config.cond_type,
        outside_embed=config.outside_embed,
        time_features=config.time_features,
    )
  else:
    raise NotImplementedError()
