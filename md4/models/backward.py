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

"""Classifier implementation."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from md4.networks import sharded_transformer
from md4.networks import transformer
from md4.networks import unet
from md4.networks import uvit


def get_timestep_embedding(timesteps, embedding_dim, dtype="float"):
    """Build sinusoidal embeddings."""

    assert embedding_dim > 2
    # timesteps: [bs]
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype="float32") * -emb)
    emb = timesteps.astype("float32")[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, jnp.array(0, dtype), ((0, 0, 0), (0, 1, 0)))
    # ret: [bs, embedding_dim]
    return emb


class CondEmbedding(nn.Module):
    """Time and cond embeddings."""

    embedding_dim: int = 256

    @nn.compact
    def __call__(self, t, cond=None):
        # t: [bs]
        n_embd = self.embedding_dim
        temb = get_timestep_embedding(t, n_embd)
        if cond is None:
            cond = temb
        else:
            cond = jnp.concatenate([temb, cond], axis=-1)
        cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense0")(cond))
        cond = nn.Dense(n_embd)(cond)
        return cond


class UNet5DWrapper(nn.Module):
    """5D to 5D UNet wrapper."""

    feature_dim: int = 128
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    output_channels: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, z, cond=None, train=False):
        # [bs, h, w, c, d or |V|] -> [bs, h, w, c, d or |V|]
        # Flatten the last two dimensions to pass to UNet
        h = z.reshape(list(z.shape)[:-2] + [-1])

        if self.n_dit_layers > 0:
            h = uvit.UNet(
                d_channels=self.feature_dim,
                n_layers=self.n_layers,
                n_dit_layers=self.n_dit_layers,
                dit_num_heads=self.dit_num_heads,
                dit_hidden_size=self.dit_hidden_size,
                ch_mult=self.ch_mult,
                output_channels=self.output_channels * z.shape[-2],
                dropout_rate=self.dropout_rate,
            )(h, cond=cond, train=train)
        else:
            h = unet.UNet(
                d_channels=self.feature_dim,
                n_layers=self.n_layers,
                output_channels=self.output_channels * z.shape[-2],
                dropout_rate=self.dropout_rate,
            )(h, cond=cond, train=train)

        # ret: [bs, h, w, c, output_channels]
        return h.reshape(list(z.shape)[:-1] + [self.output_channels])


class DiscreteClassifier(nn.Module):
    """Discrete input classifier implementation."""

    n_layers: int = 12
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    feature_dim: int = 64
    num_heads: int = 12
    n_kv_heads: int = 12
    vocab_size: int = 1000
    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    outside_embed: bool = False
    model_sharding: bool = False
    multiple_of: int = 64

    @nn.compact
    def __call__(self, z, t=None, cond=None, train=False):
        if t is not None:
            # z: [bs, seq_len] or [bs, h, w, c]
            assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
            t = t * jnp.ones(z.shape[0])  # ensure t is a vector
            cond = CondEmbedding(self.feature_dim)(t * 1000, cond=cond)

        if z.ndim == 2:
            if self.outside_embed:
                z = nn.Embed(self.vocab_size + 1, self.feature_dim)(z)
            if self.model_sharding:
                args = sharded_transformer.ModelArgs(
                    dim=self.feature_dim * self.num_heads,
                    n_layers=self.n_layers,
                    n_heads=self.num_heads,
                    n_kv_heads=self.n_kv_heads,
                    output_channels=self.vocab_size,
                    multiple_of=self.multiple_of,
                    dropout_rate=self.dropout_rate,
                    depth_scaled_init=self.depth_scaled_init,
                    mlp_type=self.mlp_type,
                    cond_type=self.cond_type,
                    embed_input=not self.outside_embed,
                    n_embed_classes=self.vocab_size + 1,
                    use_attn_dropout=self.use_attn_dropout,
                )
                # [bs, seq_len] -> [bs, seq_len, |V|]
                net = sharded_transformer.Transformer(args)
            else:
                args = transformer.ModelArgs(
                    dim=self.feature_dim * self.num_heads,
                    n_layers=self.n_layers,
                    n_heads=self.num_heads,
                    n_kv_heads=self.n_kv_heads,
                    output_channels=self.vocab_size,
                    multiple_of=self.multiple_of,
                    dropout_rate=self.dropout_rate,
                    depth_scaled_init=self.depth_scaled_init,
                    mlp_type=self.mlp_type,
                    cond_type=self.cond_type,
                    embed_input=not self.outside_embed,
                    n_embed_classes=self.vocab_size + 1,
                )
                # [bs, seq_len] -> [bs, seq_len, |V|]
                net = transformer.Transformer(args)
            logits = net(z, cond=cond, train=train)
        elif z.ndim == 4:
            z = nn.Embed(self.vocab_size + 1, self.feature_dim)(z)

            # [bs, h, w, c, d] -> [bs, h, w, c, |V|]
            net = UNet5DWrapper(
                feature_dim=self.feature_dim,
                n_layers=self.n_layers,
                n_dit_layers=self.n_dit_layers,
                dit_num_heads=self.dit_num_heads,
                dit_hidden_size=self.dit_hidden_size,
                ch_mult=self.ch_mult,
                output_channels=self.vocab_size,
                dropout_rate=self.dropout_rate,
            )
            logits = net(z, cond=cond, train=train)
        else:
            raise NotImplementedError()

        return logits, {}
