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

"""Simplified masked diffusion (MD4)."""

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from md4 import binary_search
from md4 import utils
from md4.models import backward


tfd = tfp.distributions


class MaskingSchedule(nn.Module):
    """Masking noise schedule."""

    data_shape: tuple[int, ...]
    schedule_fn_type: str = "cosine"
    eps: float = 1e-4

    def __call__(self, t):
        # return logSNR
        return jnp.log(self.alpha(t) / (1.0 - self.alpha(t)))

    def _dalpha(self, t):
        if self.schedule_fn_type == "cosine":
            return -math.pi / 2.0 * jax.lax.sin(math.pi / 2.0 * (1.0 - t))
        elif self.schedule_fn_type == "linear":
            return -jnp.ones_like(t)
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return -exponent * t ** (exponent - 1.0)
        else:
            raise NotImplementedError()

    def dalpha(self, t):
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def _alpha(self, t):
        if self.schedule_fn_type == "linear":
            return 1.0 - t
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return 1.0 - t**exponent
        elif self.schedule_fn_type == "cosine":
            return 1.0 - jax.lax.cos(math.pi / 2.0 * (1.0 - t))
        else:
            raise NotImplementedError()

    def alpha(self, t):
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps

    def dgamma_times_alpha(self, t):
        return self.dalpha(t) / (1.0 - self.alpha(t))

class SimpleMLP(nn.Module):
    """
    A simple 3-layer Multi-Layer Perceptron (MLP).

    Attributes:
        features: A sequence of integers, where each integer is the number of
                  neurons in a layer. The length of the sequence determines
                  the number of layers.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The output of the model.
        """
        # The first two layers will have a ReLU activation function.
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(features=feat, name=f"cond_dense_{i}")(x)
            x = nn.swish(x)
        
        # The final layer is the output layer and typically doesn't have an
        # activation function applied here (it might be applied in the loss function).
        x = nn.Dense(features=self.features[-1], name="cond_dense_out")(x)
        return x

class MD4(nn.Module):
    """Simplified masked discrete diffusion model."""

    data_shape: tuple[int, ...]
    cont_time: bool = False
    timesteps: int = 1000
    feature_dim: int = 128
    num_heads: int = 12
    antithetic_time_sampling: bool = True
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    vocab_size: int = 256
    noise_schedule_type: str = "linear"
    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    outside_embed: bool = False
    # time_features: t or none
    time_features: str = "t"
    classes: int = 10 + 1  # image classes
    sampler: str = "analytic"
    # uniform, cosine
    sampling_grid: str = "cosine"
    topp: float = 0.98
    model_sharding: bool = False
    fingerprint_dim: int = 0

    def setup(self):
        self.noise_schedule = MaskingSchedule(self.data_shape, self.noise_schedule_type)

        if self.classes > 0:
            self.cond_embeddings = nn.Embed(self.classes, self.feature_dim)
        if self.fingerprint_dim > 0:
            self.cond_embeddings = SimpleMLP(features=[self.fingerprint_dim // 2, self.feature_dim, self.feature_dim])

        self.classifier = backward.DiscreteClassifier(
            n_layers=self.n_layers,
            n_dit_layers=self.n_dit_layers,
            dit_num_heads=self.dit_num_heads,
            dit_hidden_size=self.dit_hidden_size,
            ch_mult=self.ch_mult,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            mlp_type=self.mlp_type,
            depth_scaled_init=self.depth_scaled_init,
            cond_type=self.cond_type,
            outside_embed=self.outside_embed,
            model_sharding=self.model_sharding,
        )

    def forward_sample(self, x, t):
        t = utils.reverse_broadcast(t, x.ndim)
        a = self.noise_schedule.alpha(t)
        un_mask = jax.random.bernoulli(self.make_rng("sample"), a, x.shape)
        # MASK = vocab_size
        return jnp.where(un_mask, x, self.vocab_size)

    def prior_sample(self, batch_size):
        return self.vocab_size * jnp.ones(
            [batch_size] + list(self.data_shape), dtype="int32"
        )

    def get_cond_embedding(self, conditioning):
        if conditioning is not None:
            return self.cond_embeddings(conditioning)
        return None

    def predict_x(self, zt, t, cond=None, train=False):
        t = None if self.time_features == "none" else t
        return self.classifier(zt, t=t, cond=cond, train=train)

    def visualize_classifier(self, x, t, conditioning=None):
        # if it's image, x: [bs, h, w, c]
        # if it's text, x: [bs, seq_len]
        cond = self.get_cond_embedding(conditioning)
        # t: []
        # if it's image, zt: [bs, h, w, c]
        # if it's text, zt: [bs, seq_len]
        zt = self.forward_sample(x, t)
        # logits: [bs, h, w, c, vocab_size] for images
        # [bs, seq_len, vocab_size] for text
        logits, _ = self.predict_x(zt, t, cond=cond)
        n_indep_axes = logits.ndim - 2
        dist = tfd.Independent(tfd.Categorical(logits=logits), n_indep_axes)
        return dist

    def encode(self, x, conditioning=None):
        del conditioning
        return x

    def decode(self, z0, conditioning=None):
        # Remove any mask tokens left in the last step of sampling.
        masked = z0 == self.vocab_size
        z0_cliped = jnp.where(masked, jnp.zeros_like(z0), z0)
        masked = masked[..., None]
        cond = self.get_cond_embedding(conditioning)
        logits, _ = self.predict_x(z0, jnp.array(0.0), cond=cond)
        probs = jnp.where(
            masked,
            nn.softmax(logits, axis=-1),
            jax.nn.one_hot(z0_cliped, self.vocab_size),
        )
        n_indep_axes = probs.ndim - 2
        dist = tfd.Independent(tfd.Categorical(probs=probs), n_indep_axes)
        return dist.mode().astype("int32")

    def recon_loss(self):
        """The reconstruction loss measures the gap in the first step."""
        alpha_t1 = self.noise_schedule.alpha(0.0)
        loss_recon = (
            jnp.prod(jnp.array(self.data_shape))
            * (1.0 - alpha_t1)
            * jnp.log(self.vocab_size)
        )
        return loss_recon

    def latent_loss(self):
        # negligible
        return jnp.array(0.0)

    def diffusion_loss(self, t, x, cond=None, train=False):
        if not self.cont_time:
            # discretize time steps
            t = (jnp.floor(t * self.timesteps) + 1) / self.timesteps

        # sample z_t
        zt = self.forward_sample(x, t)
        logits, _ = self.predict_x(zt, t, cond=cond, train=train)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        one_hot_x = jax.nn.one_hot(x, self.vocab_size)
        neg_cross_ent = one_hot_x * log_p
        neg_cross_ent = jnp.where(one_hot_x, neg_cross_ent, 0.0)
        neg_cross_ent = jnp.sum(neg_cross_ent, axis=-1)
        mask = (zt == self.vocab_size).astype("float32")

        remaining_axis = list(range(x.ndim)[1:])
        # masked_neg_cross_ent: [bs]
        masked_neg_cross_ent = jnp.sum(mask * neg_cross_ent, remaining_axis)

        if not self.cont_time:
            # loss for finite depth T, i.e. discrete time
            s = t - (1.0 / self.timesteps)
            gt = self.noise_schedule(t)
            gs = self.noise_schedule(s)
            loss_diff = (
                self.timesteps
                * jnp.expm1(gt - gs)
                * self.noise_schedule.alpha(s)
                * masked_neg_cross_ent
            )
        else:
            # cont-time loss
            loss_diff = self.noise_schedule.dgamma_times_alpha(t) * masked_neg_cross_ent

        # loss_diff: [bs]
        return loss_diff

    @nn.compact
    def __call__(self, x, cond=None, train=False):
        bs = x.shape[0]
        cond = self.get_cond_embedding(cond)

        # 1. RECONSTRUCTION LOSS: []
        # add noise and reconstruct
        loss_recon = self.recon_loss()

        # 2. LATENT LOSS: []
        loss_prior = self.latent_loss()

        # 3. DIFFUSION LOSS: [bs]
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=[bs])

        loss_diff = self.diffusion_loss(t, x, cond=cond, train=train).mean()
        loss = loss_diff + loss_prior + loss_recon

        model_stats = {
            "loss": loss,
            "loss_diff": loss_diff,
            "loss_prior": loss_prior,
            "loss_recon": loss_recon,
        }
        model_stats = utils.loss2bpt(model_stats, self.data_shape)
        return model_stats

    def get_sampling_grid(self, i, timesteps):
        t = (timesteps - i) / timesteps
        s = t - 1 / timesteps
        if self.sampling_grid == "cosine":
            t = jnp.cos(math.pi / 2.0 * (1.0 - t))
            s = jnp.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

    def ancestral_sample_step(self, rng, i, timesteps, zt, conditioning=None):
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_body)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def topp_sample_step(self, rng, i, timesteps, zt, conditioning=None, topp=0.98):
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        logits = binary_search.topp_mask(logits, topp, replace_val=jnp.array(-1e7))
        # mean_preds: [bs, ..., vocab]
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_body)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def mean_sample_step(self, rng, i, timesteps, zt, conditioning=None):
        # Ancestral sampling done in two steps -- tends to be worse than one-step
        # implementation in ancestral_sample_step. See App. G of
        # https://arxiv.org/abs/2406.04329.
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)

        rng_body, rng = jax.random.split(rng_body)
        z0 = tfd.Categorical(logits=logits).sample(seed=rng_body)

        rng_body, _ = jax.random.split(rng)
        unmask = jax.random.bernoulli(rng_body, unmask_prob, zt.shape)

        to_unmask = jnp.where(unmask, z0, zt)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def sample_step(self, rng, i, timesteps, zt, conditioning=None, topp=None):
        if self.sampler == "ancestral":
            return self.ancestral_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        elif self.sampler == "topp":
            topp = self.topp if topp is None else topp
            return self.topp_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning, topp=topp
            )
        elif self.sampler == "mean":
            return self.mean_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        else:
            raise NotImplementedError()
