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

"""Generalized state-dependent masked diffusion (GenMD4)."""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from md4 import utils
from md4.models import backward


tfd = tfp.distributions


class LearnableVecMaskingSchedule(nn.Module):
  """Learnable vector-valued masking schedule for GenMD4."""

  data_shape: tuple[int, ...]
  schedule_fn_type: str = 'poly'
  vocab_size: int = 256
  eps: float = 1e-4
  power_init: float = 1.0

  def setup(self):
    if self.schedule_fn_type == 'poly':
      w_init = jnp.log(jnp.exp(self.power_init) - 1.0)
      self.w = self.param('w', utils.constant_init(w_init), [self.vocab_size])
      # Reduce to MD4 with a shared scalar schedule:
      # self.w = self.param('w', utils.constant_init(w_init), [])
      # self.power = jnp.tile(nn.softplus(self.w)[..., None], [self.vocab_size])
      self.power = nn.softplus(self.w)
    else:
      raise NotImplementedError()

  def __call__(self, t):
    # return logSNR
    return jnp.log(self.alpha(t) / (1.0 - self.alpha(t)))

  def dalpha(self, t):
    if self.schedule_fn_type == 'poly':
      # ret: [..., |V|]
      return (
          -(1.0 - self.eps)
          * self.power
          * jnp.array(t)[..., None] ** (self.power - 1.0)
      )
    else:
      raise NotImplementedError()

  def alpha(self, t):
    if self.schedule_fn_type == 'poly':
      # instead of offsetting alpha_0 by eps as in MD4 class, we set alpha_0=1
      # and use a small non-zero t1 to avoid numerical issues, this gives a
      # nicer form of dgamma_times_alpha which is -w/t for a polynomial
      # schedule 1 - t**w
      # ret: [..., |V|]
      return 1.0 - (1.0 - self.eps) * jnp.array(t)[..., None] ** self.power
    else:
      raise NotImplementedError()

  def dgamma_times_alpha(self, t):
    # ret: [..., |V|]
    return -self.power / jnp.array(t)[..., None]


class GenMD4(nn.Module):
  """Generalized state-Dependent masked discrete diffusion model."""

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
  noise_schedule_type: str = 'poly'
  power_init: float = 1.0
  t1: float = 1e-3
  dropout_rate: float = 0.0
  use_attn_dropout: bool = True
  mlp_type: str = 'swiglu'
  depth_scaled_init: bool = False
  cond_type: str = 'adaln'
  outside_embed: bool = False
  # time_features: t or none
  time_features: str = 't'
  classes: int = 10 + 1  # image classes

  def setup(self):
    self.noise_schedule = LearnableVecMaskingSchedule(
        self.data_shape,
        schedule_fn_type=self.noise_schedule_type,
        vocab_size=self.vocab_size,
        power_init=self.power_init,
    )

    if self.classes > 0:
      self.cond_embeddings = nn.Embed(self.classes, self.feature_dim)
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
    )

  def forward_sample(self, x, t):
    t = utils.reverse_broadcast(t, x.ndim)
    # alpha_t: [bs, 1, |V|] or [bs, 1, 1, 1, |V|]
    a = self.noise_schedule.alpha(t)
    # un_mask_p: [bs, seq_len] or [bs, h, w, c]
    un_mask_p = jnp.sum(a * nn.one_hot(x, self.vocab_size), axis=-1)
    un_mask = jax.random.bernoulli(self.make_rng('sample'), un_mask_p, x.shape)
    # MASK = vocab_size
    return jnp.where(un_mask, x, self.vocab_size)

  def prior_sample(self, batch_size):
    return self.vocab_size * jnp.ones(
        [batch_size] + list(self.data_shape), dtype='int32'
    )

  def get_cond_embedding(self, conditioning):
    if conditioning is not None:
      return self.cond_embeddings(conditioning)
    return None

  def predict_x(self, zt, t, cond=None, train=False):
    t = None if self.time_features == 'none' else t
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

  def recon_loss(self, x):
    """The reconstruction loss measures the gap in the first step."""
    assert self.noise_schedule_type == 'poly'
    eps = self.noise_schedule.eps
    # w: [|V|]
    w = self.noise_schedule.power
    # w_x: [bs, seq_len] or [bs, h, w, c]
    w_x = jnp.sum(w * nn.one_hot(x, self.vocab_size), axis=-1)
    t = jnp.array(self.t1)
    # wlogt_x: [bs, seq_len] or [bs, h, w, c]
    wlogt_x = w_x * jnp.log(t)
    # wlogt: [|V|]
    wlogt = w * jnp.log(t)
    remaining_axis = list(range(x.ndim)[1:])
    # loss_recon: [bs, seq_len] or [bs, h, w, c]
    loss_recon = (
        -(1 - eps) * jnp.exp(wlogt_x) * (wlogt_x - nn.logsumexp(wlogt, -1))
    ).sum(remaining_axis)
    return loss_recon

  def latent_loss(self):
    # negligible
    return jnp.array(0.0)

  def diffusion_loss(self, t, x, cond=None, train=False):
    assert self.cont_time

    # sample z_t
    zt = self.forward_sample(x, t)
    logits, _ = self.predict_x(zt, t, cond=cond, train=train)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    one_hot_x = jax.nn.one_hot(x, self.vocab_size)
    neg_cross_ent = one_hot_x * log_p
    neg_cross_ent = jnp.where(one_hot_x, neg_cross_ent, 0.0)
    neg_cross_ent = jnp.sum(neg_cross_ent, axis=-1, keepdims=True)
    integrand = (neg_cross_ent + 1.0) * one_hot_x - jnp.exp(log_p)
    mask = (zt == self.vocab_size).astype('float')

    remaining_axis = list(range(x.ndim)[1:])
    # masked_neg_cross_ent: [bs, |V|]
    masked_neg_cross_ent = jnp.sum(mask[..., None] * integrand, remaining_axis)

    # cont-time loss
    loss_diff = (
        self.noise_schedule.dgamma_times_alpha(t) * masked_neg_cross_ent
    ).sum(axis=-1)

    # loss_diff: [bs]
    return loss_diff, zt

  def reinforce_loss(self, t, x, zt_1, zt_2, loss_diff_1, loss_diff_2):
    assert self.noise_schedule_type == 'poly'
    eps = self.noise_schedule.eps
    # w: [|V|]
    w = self.noise_schedule.power
    # w_x: [bs, seq_len] or [bs, h, w, c]
    w_x = jnp.sum(w * nn.one_hot(x, self.vocab_size), axis=-1)
    # t: [bs, 1] or [bs, 1, 1, 1]
    t = utils.reverse_broadcast(t, x.ndim)
    # alpha_t_x: [bs, seq_len] or [bs, h, w, c]
    alpha_t_x = 1.0 - (1.0 - eps) * t**w_x
    # log_q_mask = jnp.log(1.0 - alpha_t_x)
    log_q_mask = jnp.log(1.0 - eps) + w_x * jnp.log(t)
    log_q_unmask = jnp.log(alpha_t_x)
    log_q1 = jnp.where(zt_1 == self.vocab_size, log_q_mask, log_q_unmask)
    log_q2 = jnp.where(zt_2 == self.vocab_size, log_q_mask, log_q_unmask)
    remaining_axis = list(range(x.ndim)[1:])
    rloo_1 = (
        0.5
        * jax.lax.stop_gradient(loss_diff_1 - loss_diff_2)
        * (log_q1.sum(remaining_axis) - log_q2.sum(remaining_axis))
    )
    return rloo_1

  @nn.compact
  def __call__(self, x, cond=None, train=False):
    bs = x.shape[0]
    cond = self.get_cond_embedding(cond)

    # 1. RECONSTRUCTION LOSS: []
    # add noise and reconstruct
    loss_recon = self.recon_loss(x).mean()

    # 2. LATENT LOSS: []
    loss_prior = self.latent_loss()

    # 3. DIFFUSION LOSS: [bs]
    # sample time steps
    rng1 = self.make_rng('sample')
    if self.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
    else:
      t = jax.random.uniform(rng1, shape=[bs])
    # rescale t to be in [t1, 1.0]
    t = (1 - self.t1) * t + self.t1

    loss_diff_1, zt_1 = self.diffusion_loss(t, x, cond=cond, train=train)
    loss_diff_2, zt_2 = self.diffusion_loss(t, x, cond=cond, train=train)
    rloo_1 = self.reinforce_loss(t, x, zt_1, zt_2, loss_diff_1, loss_diff_2)
    loss_diff = 0.5 * (loss_diff_1 + loss_diff_2)
    loss_diff_sg = loss_diff + rloo_1

    # surrogate loss that includes the reinforce term
    loss = loss_diff_sg.mean() + loss_prior + loss_recon
    loss_diff = loss_diff.mean()
    # negative elbo
    loss_nelbo = loss_diff + loss_prior + loss_recon

    model_stats = {
        'loss': loss,
        'loss_nelbo': loss_nelbo,
        'loss_diff': loss_diff,
        'loss_prior': loss_prior,
        'loss_recon': loss_recon,
        'power_max': self.noise_schedule.power.max(),
        'power_min': self.noise_schedule.power.min(),
        'power_avg': self.noise_schedule.power.mean(),
    }
    model_stats = utils.loss2bpt(model_stats, self.data_shape)
    return model_stats
