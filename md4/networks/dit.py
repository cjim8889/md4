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

"""DiT architecture.

Jax implementation of https://arxiv.org/abs/2212.09748, based on PyTorch
implementation https://github.com/facebookresearch/DiT/blob/main/models.py.
"""

import math
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from md4.networks import transformer

# pylint: disable=missing-class-docstring


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    flatten: bool = True
    use_bias: bool = True

    def setup(self):
        self.proj = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID",
            use_bias=self.use_bias,
        )

    def __call__(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    out_features: int
    hidden_features: int
    act: Any = nn.gelu
    use_bias: bool = True
    dropout_rate: float = 0.0

    def setup(self):
        self.fc1 = nn.Dense(self.hidden_features, use_bias=self.use_bias)
        self.drop1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Dense(self.out_features, use_bias=self.use_bias)
        self.drop2 = nn.Dropout(self.dropout_rate)

    def __call__(self, x, train=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, deterministic=not train)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=not train)
        return x


class Attention(nn.Module):
    dim: int
    n_heads: int
    n_kv_heads: int | None = None
    dropout_rate: float = 0.0
    qkv_bias: bool = False

    def setup(self):
        self._n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        assert self.n_heads % self._n_kv_heads == 0
        self.n_rep = self.n_heads // self._n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Dense(self.n_heads * self.head_dim, use_bias=self.qkv_bias)
        self.wk = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
        self.wv = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
        self.wo = nn.Dense(self.dim, use_bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_rate)
        # self.resid_dropout = nn.Dropout(self.dropout_rate)
        self.resid_dropout = transformer.Dropout1d(self.dropout_rate)

    def __call__(self, x, train=False):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)

        # grouped multiquery attention: expand out keys and values
        xk = transformer.repeat_kv(xk, self.n_rep)
        xv = transformer.repeat_kv(xv, self.n_rep)

        # make heads into a batch dimension
        xq = xq.swapaxes(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.swapaxes(1, 2)
        xv = xv.swapaxes(1, 2)

        scores = jnp.matmul(xq, xk.swapaxes(2, 3)) / math.sqrt(self.head_dim)
        scores = nn.softmax(scores, axis=-1)
        scores = self.attn_dropout(scores, deterministic=not train)
        output = jnp.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.swapaxes(1, 2).reshape(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output, deterministic=not train)
        return output


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

    def setup(self):
        self.attn = Attention(
            self.hidden_size,
            self.num_heads,
            dropout_rate=self.dropout_rate,
            qkv_bias=True,
        )
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = Mlp(
            out_features=self.hidden_size,
            hidden_features=mlp_hidden_dim,
            act=nn.gelu,
            dropout_rate=self.dropout_rate,
        )

    @nn.compact
    def __call__(self, x, cond=None, train=False):
        if cond is not None:
            adaln_modulation = nn.Sequential(
                [
                    nn.swish,
                    nn.Dense(
                        6 * self.hidden_size,
                        kernel_init=nn.zeros_init(),
                        bias_init=nn.zeros_init(),
                    ),
                ]
            )
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
                adaln_modulation(cond)[:, None, :], 6, axis=-1
            )
            norm1 = nn.LayerNorm(use_bias=False, use_scale=False)
            norm2 = nn.LayerNorm(use_bias=False, use_scale=False)
            x = x + gate_msa * self.attn(
                modulate(norm1(x), shift_msa, scale_msa), train=train
            )
            x = x + gate_mlp * self.mlp(
                modulate(norm2(x), shift_mlp, scale_mlp), train=train
            )
        else:
            x = x + self.attn(nn.RMSNorm()(x), train=train)
            x = x + self.mlp(nn.RMSNorm()(x), train=train)
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.linear = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=nn.zeros_init(),
            bias_init=nn.zeros_init(),
        )

    @nn.compact
    def __call__(self, x, cond=None):
        if cond is not None:
            adaln_modulation = nn.Sequential(
                [
                    nn.swish,
                    nn.Dense(
                        2 * self.hidden_size,
                        kernel_init=nn.zeros_init(),
                        bias_init=nn.zeros_init(),
                    ),
                ]
            )
            shift, scale = jnp.split(adaln_modulation(cond)[:, None, :], 2, axis=-1)
            norm_final = nn.LayerNorm(use_bias=False, use_scale=False)
            x = modulate(norm_final(x), shift, scale)
        else:
            x = nn.RMSNorm()(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    img_size: int
    patch_size: int
    in_channels: int
    out_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

    def setup(self):
        self.x_embedder = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.hidden_size,
            use_bias=True,
        )
        self.grid_size = self.img_size // self.patch_size
        num_patches = self.grid_size * self.grid_size
        self.pos_embed = self.param(
            "pos_embed",
            lambda k, s: get_2d_sincos_pos_embed(s[-1], int(num_patches**0.5)),
            [num_patches, self.hidden_size],
        )
        self.blocks = [
            DiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.depth)
        ]
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

    def __call__(self, x, cond=None, train=False):
        c = x.shape[-1]
        p = self.patch_size
        grid_size = self.grid_size
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / p ** 2
        for block in self.blocks:
            x = block(x, cond=cond, train=train)  # (N, T, D)
        x = self.final_layer(x, cond=cond)  # (N, T, p ** 2 * c)
        x = x.reshape(-1, grid_size, grid_size, p, p, c)
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        x = x.reshape(-1, grid_size * p, grid_size * p, c)  # (N, H, W, c)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2D sin-cos position embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Gets 2D sin-cos position embedding from grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Gets 1D sin-cos position embedding from grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
