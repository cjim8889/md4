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
from md4.models import backward
from md4.networks.adapters import FingerprintAdapter, SimpleMLP
from md4.utils import utils

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

class MD4(nn.Module):
    """Simplified masked discrete diffusion model."""

    data_shape: tuple[int, ...]
    cont_time: bool = False
    timesteps: int = 1000
    feature_dim: int = 128
    num_heads: int = 12
    n_kv_heads: int = 12
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
    fingerprint_adapter: bool = False
    only_adapter: bool = False
    raw_fingerprint_dim: int = 0
    atom_type_size: int = 0
    fingerprint_mlp_layers: Sequence[int] = ()
    multiple_of: int = 64
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.noise_schedule = MaskingSchedule(self.data_shape, self.noise_schedule_type)

        if self.classes > 0:
            self.cond_embeddings = nn.Embed(self.classes, self.feature_dim, dtype=jnp.float32, param_dtype=self.param_dtype)
        if self.fingerprint_dim > 0:
            if self.fingerprint_adapter:
                self.fp_adapter = FingerprintAdapter(
                    raw_fingerprint_dim=self.raw_fingerprint_dim,
                    fingerprint_dim=self.fingerprint_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )

            # Use configurable layers if provided, otherwise use default
            if self.fingerprint_mlp_layers:
                mlp_features = list(self.fingerprint_mlp_layers)
            else:
                mlp_features = [self.fingerprint_dim // 2, self.feature_dim * 2, self.feature_dim, self.feature_dim]
            
            self.cond_embeddings = SimpleMLP(
                features=mlp_features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.atom_type_size > 0:
            self.atom_embeddings = nn.Embed(self.atom_type_size, self.feature_dim, dtype=jnp.float32, param_dtype=self.param_dtype)
            self.atom_embeddings_agg = nn.Dense(features=self.feature_dim, name="atom_embeddings_agg", dtype=self.dtype, param_dtype=self.param_dtype)

        self.classifier = backward.DiscreteClassifier(
            n_layers=self.n_layers,
            n_dit_layers=self.n_dit_layers,
            dit_num_heads=self.dit_num_heads,
            dit_hidden_size=self.dit_hidden_size,
            ch_mult=self.ch_mult,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            mlp_type=self.mlp_type,
            depth_scaled_init=self.depth_scaled_init,
            cond_type=self.cond_type,
            outside_embed=self.outside_embed,
            model_sharding=self.model_sharding,
            multiple_of=self.multiple_of,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
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
    
    def conditional_sample(self, tokens):
        """
        Given tokens shaped (..., L), find the first occurrence of token id 3
        along the last dimension and set all subsequent positions to
        self.vocab_size (mask token). If a sequence has no 3, it is unchanged.

        Args:
            tokens: jnp.ndarray[int], shape (..., L)

        Returns:
            jnp.ndarray[int], same shape as tokens, with positions strictly
            after the first 3 replaced by self.vocab_size.
        """
        # Positions where token equals 3
        is_stop = (tokens == 3)

        # Inclusive mask that becomes True at and after the first 3
        hit_inclusive = (jnp.cumsum(is_stop.astype(jnp.int32), axis=-1) > 0)

        # We only want positions strictly after the first 3: shift right by 1
        zeros = jnp.zeros_like(hit_inclusive[..., :1])
        strictly_after = jnp.concatenate([zeros, hit_inclusive[..., :-1]], axis=-1)

        mask_val = jnp.asarray(self.vocab_size, dtype=tokens.dtype)
        return jnp.where(strictly_after, mask_val, tokens).astype(jnp.int32)

    def get_cond_embedding(self, conditioning):
        fp_logits = None
        if conditioning is not None:
            if isinstance(conditioning, dict):
                _cond = conditioning["fingerprint"]
                if self.fingerprint_adapter:
                    # Convert raw fingerprint to the desired dimension
                    if conditioning["fingerprint"].shape[1] != self.raw_fingerprint_dim:
                        # Print stack trace for debugging
                        raise ValueError(
                            f"Expected fingerprint shape {self.raw_fingerprint_dim}, got {conditioning['fingerprint'].shape[1]}"
                        )
                    # _cond = nn.sigmoid(self.cond_conversion(conditioning["fingerprint"]))
                    _cond, fp_logits = self.fp_adapter(conditioning["fingerprint"])

                if "atom_types" in conditioning:
                    atom_conditioning = self.atom_embeddings(conditioning["atom_types"])
                    atom_conditioning = jax.vmap(self.atom_embeddings_agg)(atom_conditioning)
                    atom_conditioning = nn.swish(atom_conditioning.astype(jnp.float32)).astype(self.dtype)

                    if atom_conditioning.ndim == 2:
                        atom_conditioning = jnp.sum(atom_conditioning, axis=0)
                    elif atom_conditioning.ndim == 3:
                        atom_conditioning = jnp.sum(atom_conditioning, axis=1)
                    else:
                        raise ValueError("Atom conditioning has invalid shape")

                    # Ensure consistent dtypes for concatenation
                    _cond = _cond.astype(self.dtype)
                    atom_conditioning = atom_conditioning.astype(self.dtype)
                    _cond = jnp.concat([_cond, atom_conditioning], axis=-1)

                return self.cond_embeddings(_cond), fp_logits

            return self.cond_embeddings(conditioning), fp_logits
        return None, fp_logits

    def predict_x(self, zt, t, cond=None, train=False):
        t = None if self.time_features == "none" else t
        return self.classifier(zt, t=t, cond=cond, train=train)

    def visualize_classifier(self, x, t, conditioning=None):
        # if it's image, x: [bs, h, w, c]
        # if it's text, x: [bs, seq_len]
        cond, _ = self.get_cond_embedding(conditioning)
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
        cond, _ = self.get_cond_embedding(conditioning)
        logits, _ = self.predict_x(z0, jnp.array(0.0), cond=cond)
        probs = jnp.where(
            masked,
            nn.softmax(logits, axis=-1),
            jax.nn.one_hot(z0_cliped, self.vocab_size),
        )
        n_indep_axes = probs.ndim - 2
        dist = tfd.Independent(tfd.Categorical(probs=probs), n_indep_axes)
        return dist.mode().astype(jnp.int32)

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
    
    def fp_bce_loss(self, logits, labels):
        """Binary Cross Entropy loss."""
        labels = labels.astype(self.dtype)
        log_p = jax.nn.log_sigmoid(logits)
        # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
        log_not_p = jax.nn.log_sigmoid(-logits)
        return -labels * log_p - (1.0 - labels) * log_not_p
    
    def fp_focal_loss(self, logits, labels, alpha=1.5, gamma=2.0):
        """
        Focal loss implementation.
        
        L(y, p̂) = -α y (1 - p̂)^γ log(p̂) - (1 - y) p̂^γ log(1 - p̂)
        
        Args:
            logits: Model predictions (before sigmoid)
            labels: Ground truth binary labels
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        labels = labels.astype(self.dtype)
        probas = jax.nn.sigmoid(logits.astype(jnp.float32))
        
        # Calculate focal weights
        pt = labels * probas + (1.0 - labels) * (1.0 - probas)
        focal_weight = jnp.power(1.0 - pt, gamma)
        
        # Calculate cross entropy
        ce = self.fp_bce_loss(logits, labels)
        
        # Apply focal loss formula  
        alpha = jnp.asarray(alpha, dtype=self.dtype)
        loss = alpha * focal_weight * ce
        return loss
    
    def diffusion_loss(self, t, x, cond=None, train=False):
        if not self.cont_time:
            # discretize time steps
            t = (jnp.floor(t * self.timesteps) + 1) / self.timesteps

        # sample z_t
        zt = self.forward_sample(x, t)
        # Ensure all loss computations are in fp32 for numerical stability
        logits, _ = self.predict_x(zt, t, cond=cond, train=train)
        logits = logits.astype(jnp.float32)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        one_hot_x = jax.nn.one_hot(x, self.vocab_size).astype(jnp.float32)
        neg_cross_ent = one_hot_x * log_p
        neg_cross_ent = jnp.where(one_hot_x, neg_cross_ent, 0.0)
        neg_cross_ent = jnp.sum(neg_cross_ent, axis=-1)
        mask = jnp.asarray(zt == self.vocab_size, dtype=jnp.float32)

        remaining_axis = list(range(x.ndim)[1:])
        # masked_neg_cross_ent: [bs]
        masked_neg_cross_ent = jnp.sum(mask * neg_cross_ent, remaining_axis)

        if not self.cont_time:
            # loss for finite depth T, i.e. discrete time
            s = t - (1.0 / self.timesteps)
            gt = self.noise_schedule(t)
            gs = self.noise_schedule(s)
            # Ensure numerical stability by casting to fp32
            loss_diff = (
                jnp.asarray(self.timesteps, dtype=jnp.float32)
                * jnp.expm1(gt - gs).astype(jnp.float32)
                * self.noise_schedule.alpha(s).astype(jnp.float32)
                * masked_neg_cross_ent
            )
        else:
            # cont-time loss
            loss_diff = (self.noise_schedule.dgamma_times_alpha(t).astype(jnp.float32) 
                        * masked_neg_cross_ent)

        # loss_diff: [bs]
        return loss_diff.astype(jnp.float32)

    @nn.compact
    def __call__(self, x, cond=None, train=False):
        bs = x.shape[0]
        cond_embedding, fp_logits = self.get_cond_embedding(cond)

        # 0. FINGERPRINT ADAPTER LOSS: []
        loss_fp = jnp.array(0.0)
        avg_bit_diff = jnp.array(0.0)
        
        if (self.fingerprint_adapter and fp_logits is not None and 
            cond is not None and isinstance(cond, dict) and "true_fingerprint" in cond):
            # Binary cross entropy loss between logits and true fingerprint
            true_fp = cond["true_fingerprint"]
            loss_fp = jnp.mean(self.fp_focal_loss(fp_logits, true_fp))
            
            # Calculate average bit differences between true_fp and predicted fingerprint
            # Convert logits to binary predictions
            pred_fp = jnp.where(nn.sigmoid(fp_logits.astype(jnp.float32)) < 0.5, 0.0, 1.0)
            bit_diff = jnp.abs(true_fp - pred_fp)
            avg_bit_diff = jnp.mean(jnp.sum(bit_diff, axis=-1))

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

        loss_diff = self.diffusion_loss(t, x, cond=cond_embedding, train=train).mean()

        if self.only_adapter:
            loss = loss_fp
        else:
            # Cast all loss components to same dtype for stable addition
            loss_diff = loss_diff.astype(jnp.float32)
            loss_prior = loss_prior.astype(jnp.float32)
            loss_recon = loss_recon.astype(jnp.float32)
            loss_fp = loss_fp.astype(jnp.float32)
            loss = loss_diff + loss_prior + loss_recon + loss_fp

        model_stats = {
            "loss": loss,
            "loss_diff": loss_diff,
            "loss_prior": loss_prior,
            "loss_recon": loss_recon,
            "loss_fp": loss_fp,
            "avg_bit_diff": avg_bit_diff,
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
        cond, _ = self.get_cond_embedding(conditioning)

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
        cond, _ = self.get_cond_embedding(conditioning)

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
        cond, _ = self.get_cond_embedding(conditioning)

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
