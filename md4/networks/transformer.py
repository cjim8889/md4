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

"""Jax implementation of LLAMA2-like Transformer.

Based on PyTorch implementation
https://github.com/karpathy/llama2.c/blob/master/model.py
"""

import dataclasses
import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


activation_map = dict(
    swiglu=nn.swish,
    geglu=nn.gelu,
    glu=nn.sigmoid,
)


@dataclasses.dataclass(unsafe_hash=True)
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    output_channels: int = 1024
    hidden_dim: Optional[int] = None
    multiple_of: int = 32  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    dropout_rate: float = 0.0
    weight_tying: bool = False
    w_init_scale: float = 1.0
    depth_scaled_init: bool = False
    # glu, geglu, swiglu
    mlp_type: str = "swiglu"
    # adaln, adaln_zero
    cond_type: str = "adaln"
    embed_input: bool = False
    n_embed_classes: int = 1024
    causal: bool = False


class RMSNorm(nn.Module):
    dim: int
    eps: float

    def setup(self):
        self.scale = self.param(
            "scale", lambda key, shape: jnp.ones(shape), (self.dim,)
        )

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x)
        return output * self.scale


def precompute_freqs_cis(dim, end, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def jax_unstack(x, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])
    ]


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # reshape xq and xk to match the complex representation
    # [bs, seq_len, n_head, head_dim // 2]
    xq_r, xq_i = jax_unstack(xq.reshape(xq.shape[:-1] + (-1, 2)), -1)
    xk_r, xk_i = jax_unstack(xk.reshape(xk.shape[:-1] + (-1, 2)), -1)

    # reshape freqs_cos and freqs_sin for broadcasting
    # [1, seq_len, 1, head_dim // 2]
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    # [bs, seq_len, n_head, head_dim // 2, 2] -> [bs, seq_len, n_head, head_dim]
    xq_out = jnp.stack([xq_out_r, xq_out_i], axis=-1).reshape(
        xq_out_r.shape[:3] + (-1,)
    )
    xk_out = jnp.stack([xk_out_r, xk_out_i], axis=-1).reshape(
        xk_out_r.shape[:3] + (-1,)
    )

    return xq_out, xk_out


def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, None, :], [1, 1, 1, n_rep, 1]).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


class Dropout1d(nn.Module):
    dropout_rate: float = 0.0

    def __call__(self, x, deterministic=True):
        if (self.dropout_rate > 0.0) and not deterministic:
            drop = jax.random.bernoulli(
                self.make_rng("dropout"),
                1 - self.dropout_rate,
                (x.shape[0], 1, x.shape[-1]),
            )
            x = x * drop / (1 - self.dropout_rate)
        return x


class Attention(nn.Module):
    dim: int
    n_heads: int
    n_kv_heads: int | None = None
    dropout_rate: float = 0.0
    causal: bool = False
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
        if self.dropout_rate > 0.0:
            self.attn_dropout = nn.Dropout(self.dropout_rate)
            self.resid_dropout = Dropout1d(self.dropout_rate)

    def __call__(self, x, freqs_cos, freqs_sin, train=False):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # make heads into a batch dimension
        xq = xq.swapaxes(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.swapaxes(1, 2)
        xv = xv.swapaxes(1, 2)

        xk_transposed = xk.swapaxes(2, 3)
        scores = jnp.matmul(xq, xk_transposed) / math.sqrt(self.head_dim)
        if self.causal:
            mask = jnp.full((1, 1, seqlen, seqlen), -jnp.inf)
            mask = jnp.triu(mask, k=1)
            scores = (
                scores + mask[:, :, :seqlen, :seqlen]
            )  # (bs, n_heads, seqlen, seqlen)
        scores = nn.softmax(scores, axis=-1)
        if self.dropout_rate > 0.0:
            scores = self.attn_dropout(scores, deterministic=not train)
        output = jnp.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.swapaxes(1, 2).reshape(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        if self.dropout_rate > 0.0:
            output = self.resid_dropout(output, deterministic=not train)
        return output


def dense_2d(dense, x):
    x2 = x.reshape(-1, x.shape[-1])  # [B*S, D]
    y2 = dense(x2)  # [B*S, H]
    return y2.reshape(*x.shape[:-1], -1)  # [B, S, H]


class FeedForward(nn.Module):
    dim: int
    multiple_of: int
    dropout_rate: float
    hidden_dim: int | None = None
    w_init_scale: float = 1.0
    mlp_type: str = "swiglu"

    def setup(self):
        multiple_of = self.multiple_of
        hidden_dim = self.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        w_init = nn.initializers.variance_scaling(
            self.w_init_scale, "fan_in", "truncated_normal"
        )
        self.w1 = nn.Dense(hidden_dim, use_bias=False, kernel_init=w_init)
        self.w2 = nn.Dense(self.dim, use_bias=False, kernel_init=w_init)
        self.w3 = nn.Dense(hidden_dim, use_bias=False, kernel_init=w_init)
        # self.dropout = nn.Dropout(self.dropout_rate)
        if self.dropout_rate > 0.0:
            self.dropout = Dropout1d(self.dropout_rate)

    def __call__(self, x, train=False):
        act = activation_map[self.mlp_type]
        u = dense_2d(self.w1, x)
        v = dense_2d(self.w3, x)        # used for *any* gated MLP; ignore if plain GELU MLP
        y = dense_2d(self.w2, act(u) * v)
        if self.dropout_rate > 0.0:
            y = self.dropout(y, deterministic=not train)
        return y


class TransformerBlock(nn.Module):
    layer_id: int
    args: ModelArgs

    def setup(self):
        args = self.args
        self.attention = Attention(
            args.dim,
            args.n_heads,
            n_kv_heads=args.n_kv_heads,
            dropout_rate=args.dropout_rate,
            causal=args.causal,
        )

        if args.depth_scaled_init:
            w_init_scale = 2.0 / args.n_layers
        else:
            w_init_scale = args.w_init_scale

        self.feed_forward = FeedForward(
            dim=args.dim,
            multiple_of=args.multiple_of,
            dropout_rate=args.dropout_rate,
            hidden_dim=args.hidden_dim,
            w_init_scale=w_init_scale,
            mlp_type=args.mlp_type,
        )

    @nn.compact
    def __call__(self, x, freqs_cos, freqs_sin, cond=None, train=False):
        if cond is not None:
            activation = activation_map[self.args.mlp_type]
            if self.args.cond_type == "adaln":
                ln = nn.Sequential(
                    [
                        # nn.swish,
                        activation,
                        nn.Dense(6 * self.args.dim, use_bias=True),
                    ]
                )
            elif self.args.cond_type == "adaln_zero":
                ln = nn.Sequential(
                    [
                        # nn.swish,
                        activation,
                        nn.Dense(
                            6 * self.args.dim,
                            use_bias=True,
                            kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros,
                        ),
                    ]
                )
            else:
                raise NotImplementedError()
            (shift_att, scale_att, gate_att, shift_mlp, scale_mlp, gate_mlp) = (
                jnp.split(ln(cond)[:, None, :], 6, axis=-1)
            )
            attention_norm = nn.LayerNorm(
                epsilon=self.args.norm_eps, use_bias=False, use_scale=False
            )
            ffn_norm = nn.LayerNorm(
                epsilon=self.args.norm_eps, use_bias=False, use_scale=False
            )
            h = x + gate_att * self.attention(
                attention_norm(x) * (scale_att + 1.0) + shift_att,
                freqs_cos,
                freqs_sin,
                train=train,
            )
            out = h + gate_mlp * self.feed_forward(
                ffn_norm(h) * (scale_mlp + 1.0) + shift_mlp, train=train
            )
        else:
            attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
            ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
            h = x + self.attention(attention_norm(x), freqs_cos, freqs_sin, train=train)
            out = h + self.feed_forward(ffn_norm(h), train=train)

        return out


class Transformer(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x, cond=None, train=False, output_channels=None):
        args = self.args
        if output_channels is None:
            output_channels = args.output_channels

        if args.embed_input:
            h = nn.Embed(args.n_embed_classes, args.dim)(x)
            if args.dropout_rate > 0.0:
                h = nn.Dropout(args.dropout_rate, deterministic=not train)(h)
        else:
            h = nn.Dense(args.dim)(x)

        seqlen = x.shape[1]
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, seqlen)

        freqs_cos = freqs_cos[:seqlen]
        freqs_sin = freqs_sin[:seqlen]

        for layer_id in range(args.n_layers):
            h = TransformerBlock(layer_id, args)(
                h, freqs_cos, freqs_sin, cond=cond, train=train
            )

        if cond is not None:
            output_norm = nn.LayerNorm(
                epsilon=args.norm_eps, use_bias=False, use_scale=False
            )
            activation = activation_map[args.mlp_type]
            if args.cond_type == "adaln":
                ln = nn.Sequential(
                    [
                        # nn.swish,
                        activation,
                        nn.Dense(2 * args.dim, use_bias=True),
                    ]
                )
            elif args.cond_type == "adaln_zero":
                ln = nn.Sequential(
                    [
                        # nn.swish,
                        activation,
                        nn.Dense(
                            2 * args.dim,
                            use_bias=True,
                            kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros,
                        ),
                    ]
                )
            else:
                raise NotImplementedError()
            shift_out, scale_out = jnp.split(ln(cond)[:, None, :], 2, axis=-1)
            logits = nn.Dense(
                output_channels, use_bias=False, kernel_init=nn.initializers.zeros
            )(output_norm(h) * (scale_out + 1) + shift_out)
        else:
            h = RMSNorm(args.dim, args.norm_eps)(h)
            logits = nn.Dense(
                features=output_channels,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
            )(h)

        return logits
