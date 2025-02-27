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

"""UViT implementation based on DiT."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from md4.networks.dit import DiT


def nearest_neighbor_upsample(x):
  bs, h, w, c = x.shape
  x = x.reshape(bs, h, 1, w, 1, c)
  x = jnp.broadcast_to(x, (bs, h, 2, w, 2, c))
  return x.reshape(bs, h * 2, w * 2, c)


class CondGroupNorm(nn.Module):
  """Conditional normalization."""

  @nn.compact
  def __call__(self, x, cond=None):
    c = x.shape[-1]
    if cond is not None:
      cond_act = nn.Dense(c * 2, kernel_init=nn.initializers.zeros)(cond)
      scale, shift = jnp.split(cond_act[:, None, None, :], 2, axis=-1)
      x = nn.GroupNorm(use_bias=False, use_scale=False)(x) * (1 + scale) + shift
    else:
      x = nn.GroupNorm()(x)
    return x


class SelfAttention(nn.Module):
  """Self attention layer in UNets."""

  num_heads: int = 1

  @nn.compact
  def __call__(self, x, cond=None):
    bs, h, w, c = x.shape
    z = CondGroupNorm()(x, cond=cond)
    z = z.reshape(bs, -1, c)

    mha = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=c,
        out_kernel_init=nn.zeros_init(),
    )
    z = mha(z, z)
    z = z.reshape(-1, h, w, c) + x
    return z


class ResBlock(nn.Module):
  """Residual block in UNets."""

  out_channels: int | None = None
  dropout_rate: float = 0.1
  resample: str | None = None

  @nn.compact
  def __call__(self, x, cond=None, train=False):
    in_channels = x.shape[-1]
    out_channels = (
        in_channels if self.out_channels is None else self.out_channels
    )
    h = CondGroupNorm()(x, cond=cond)
    h = nn.swish(h)

    if self.resample is not None:

      def updown(z):
        return {
            'up': nearest_neighbor_upsample(z),
            'down': nn.avg_pool(z, (2, 2), (2, 2)),
        }[self.resample]

      h = updown(h)
      x = updown(x)
    h = nn.Conv(out_channels, kernel_size=(3, 3))(h)
    h = CondGroupNorm()(h, cond=cond)
    h = nn.swish(h)
    h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
    h = nn.Conv(
        out_channels, kernel_size=(3, 3), kernel_init=nn.initializers.zeros
    )(h)
    if in_channels != out_channels:
      x = nn.Dense(out_channels)(x)
    return x + h


class UNet(nn.Module):
  """UNet for Diffusion."""

  d_channels: int = 128
  n_layers: int = 32
  n_dit_layers: int = 0
  dit_num_heads: int = 12
  dit_hidden_size: int = 768
  ch_mult: Sequence[int] = (1,)
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
    num_resolutions = len(self.ch_mult)

    # Linear projection of input
    h = nn.Conv(self.d_channels, kernel_size=(3, 3))(x)
    hs = [h]

    # Downsampling
    for i_level in range(num_resolutions):
      for _ in range(self.n_layers):
        h = ResBlock(
            out_channels=self.d_channels * self.ch_mult[i_level],
            dropout_rate=self.dropout_rate,
        )(h, cond=cond, train=train)
        hs.append(h)
      # Downsample
      if i_level != num_resolutions - 1:
        h = ResBlock(
            dropout_rate=self.dropout_rate,
            resample='down',
        )(h, cond=cond, train=train)
        hs.append(h)

    # Middle
    _, img_size, _, c = h.shape
    h = DiT(
        img_size=img_size,
        patch_size=2,
        in_channels=c,
        out_channels=c,
        hidden_size=self.dit_hidden_size,  # c * 2, c * 3..
        depth=self.n_dit_layers,  # 8, 12, 16, 20
        num_heads=self.dit_num_heads,  # 8, 12..
        dropout_rate=self.dropout_rate,
    )(h, cond=cond, train=train)

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      for _ in range(self.n_layers + 1):
        h = jnp.concatenate([h, hs.pop()], axis=-1)
        h = ResBlock(
            out_channels=self.d_channels * self.ch_mult[i_level],
            dropout_rate=self.dropout_rate,
        )(h, cond=cond, train=train)
      # Upsample
      if i_level != 0:
        h = ResBlock(
            dropout_rate=self.dropout_rate,
            resample='up',
        )(h, cond=cond, train=train)

    assert not hs

    # Predict noise
    h = nn.swish(CondGroupNorm()(h, cond=cond))
    h = nn.Conv(output_channels, (3, 3), kernel_init=nn.initializers.zeros)(h)

    if self.add_input:
      h += x
    return h
