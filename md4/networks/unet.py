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

"""UNet implementation.

Adapted from https://github.com/google-research/vdm/blob/main/model_vdm.py
"""

import flax.linen as nn
import jax.numpy as jnp


class SelfAttention(nn.Module):
  """Self attention layer in UNets."""

  num_heads: int = 1

  @nn.compact
  def __call__(self, x):
    _, h, w, c = x.shape
    z = nn.GroupNorm()(x)
    z = z.reshape(z.shape[0], -1, c)
    mha = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, qkv_features=c
    )
    z = mha(z, z)
    z = z.reshape(-1, h, w, c) + x
    return z


class ResBlock(nn.Module):
  """Residual block in UNets."""

  out_channels: int
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x, cond=None, train=False):
    in_channels = x.shape[-1]
    h = nn.GroupNorm()(x)
    h = nn.swish(h)
    h = nn.Conv(self.out_channels, kernel_size=(3, 3))(h)
    if cond is not None:
      cond_act = nn.Dense(
          self.out_channels,
          use_bias=False,
          kernel_init=nn.initializers.zeros,
      )(cond)
      h = h + cond_act[:, None, None, :]
    h = nn.GroupNorm()(h)
    h = nn.swish(h)
    h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
    h = nn.Conv(
        self.out_channels, kernel_size=(3, 3), kernel_init=nn.initializers.zeros
    )(h)
    if in_channels != self.out_channels:
      h = nn.Dense(self.out_channels)(x) + h
    else:
      h = x + h
    return h


class UNet(nn.Module):
  """UNet for Diffusion."""

  d_channels: int = 128
  n_layers: int = 32
  add_input: bool = False
  output_channels: int | None = None
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x, cond=None, train=False, output_channels=None):
    if output_channels is None:
      if self.output_channels is None:
        output_channels = x.shape[-1]
      else:
        output_channels = self.output_channels

    # Linear projection of input
    h = nn.Conv(self.d_channels, kernel_size=(3, 3))(x)
    hs = [h]

    # Downsampling
    for _ in range(self.n_layers):
      h = ResBlock(
          out_channels=self.d_channels, dropout_rate=self.dropout_rate
      )(h, cond, train)
      hs.append(h)

    # Middle
    h = ResBlock(out_channels=self.d_channels, dropout_rate=self.dropout_rate)(
        h, cond, train
    )
    h = SelfAttention(num_heads=1)(h)
    h = ResBlock(out_channels=self.d_channels, dropout_rate=self.dropout_rate)(
        h, cond, train
    )

    # Upsampling
    for _ in range(self.n_layers + 1):
      h = jnp.concatenate([h, hs.pop()], axis=-1)
      h = ResBlock(
          out_channels=self.d_channels, dropout_rate=self.dropout_rate
      )(h, cond, train)

    assert not hs

    # Predict noise
    h = nn.swish(nn.GroupNorm()(h))
    h = nn.Conv(output_channels, (3, 3), kernel_init=nn.initializers.zeros)(h)

    if self.add_input:
      h += x
    return h
