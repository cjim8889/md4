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
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from md4.networks import sharded_transformer
from md4.networks import transformer


def get_timestep_embedding(timesteps, embedding_dim, dtype=jnp.float32):
    """Build sinusoidal embeddings."""

    assert embedding_dim > 2
    # timesteps: [bs]
    # Compute everything in fp32 for numerical stability
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, jnp.array(0, dtype=jnp.float32), ((0, 0, 0), (0, 1, 0)))
    # Convert to target dtype at the end
    # ret: [bs, embedding_dim]
    return emb.astype(dtype)


class CondEmbedding(nn.Module):
    """Time and cond embeddings."""

    embedding_dim: int = 256
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, t, cond=None):
        # t: [bs]
        n_embd = self.embedding_dim
        temb = get_timestep_embedding(t, n_embd, dtype=self.dtype)
        if cond is None:
            cond = temb
        else:
            cond = jnp.concatenate([temb, cond], axis=-1)
        cond = nn.swish(nn.Dense(
            features=n_embd * 4, 
            name="dense0", 
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.linear.default_kernel_init, ('cond_input', 'cond_hidden')
            )
        )(cond))
        cond = nn.Dense(
            n_embd, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.linear.default_kernel_init, ('cond_hidden', 'cond_output')
            )
        )(cond)
        return cond


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
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_cross_attention: bool = False
    cross_attention_layers: Optional[int] = None
    cross_cond_proj_dim: Optional[int] = None

    @nn.compact
    def __call__(self, z, t=None, cond=None, cross_conditioning=None, train=False):
        if t is not None:
            # z: [bs, seq_len] or [bs, h, w, c]
            assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
            t = t * jnp.ones(z.shape[0])  # ensure t is a vector
            cond = CondEmbedding(self.feature_dim, dtype=self.dtype, param_dtype=self.param_dtype)(t * 1000, cond=cond)

        if z.ndim == 2:
            if self.outside_embed:
                z = nn.Embed(
                    self.vocab_size + 1, 
                    self.feature_dim, 
                    dtype=self.dtype, 
                    param_dtype=self.param_dtype,
                    embedding_init=nn.with_logical_partitioning(
                        nn.linear.default_embed_init, ('vocab', 'hidden')
                    )
                )(z)
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
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    use_cross_attention=self.use_cross_attention,
                    cross_attention_layers=self.cross_attention_layers,
                    cross_cond_proj_dim=self.cross_cond_proj_dim,
                )
                # [bs, seq_len] -> [bs, seq_len, |V|]
                net = transformer.Transformer(args)
            logits = net(z, cond=cond, cross_conditioning=cross_conditioning, train=train)
        else:
            raise NotImplementedError()

        return logits, {}
