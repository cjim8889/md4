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

"""Sampling functions."""

import functools

import jax
import jax.numpy as jnp


def get_attr(train_state, key):
  if hasattr(train_state, key):
    return getattr(train_state, key)
  else:
    return train_state[key]


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 5))
def generate(
    model,
    train_state,
    rng,
    dummy_inputs,
    conditioning=None,
    use_conditional_init: bool = False,
):
  """Generate samples from the diffusion model."""
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  variables = {
      'params': get_attr(train_state, 'ema_params'),
      **get_attr(train_state, 'state'),
  }
  rng, sub_rng = jax.random.split(rng)
  if use_conditional_init:
    # Initialize zt from provided token sequences by masking after the first stop token (id=3).
    tokens = dummy_inputs["smiles"]
    zt = model.apply(
        variables,
        tokens,
        method=model.conditional_sample,
        rngs={'sample': sub_rng}
    )
  else:
    zt = model.apply(
        variables,
        dummy_inputs.shape[0],
        method=model.prior_sample,
        rngs={'sample': sub_rng},
    )
  rng, sub_rng = jax.random.split(rng)

  def body_fn(i, zt):
    return model.apply(
        variables,
        sub_rng,
        i,
        model.timesteps,
        zt,
        conditioning=conditioning,
        method=model.sample_step,
    )

  z0 = jax.lax.fori_loop(
      lower=0, upper=model.timesteps, body_fun=body_fn, init_val=zt
  )
  return model.apply(
      variables,
      z0,
      conditioning=conditioning,
      method=model.decode,
      rngs={'sample': rng},
  )


def simple_generate(
    rng,
    train_state,
    conditioning,
    batch_size,
    model,
    dummy_inputs=None,
    use_conditional_init: bool = False,
):
  """Generate samples from the diffusion model."""
  variables = {'params': train_state.params, **train_state.state}
  rng, sub_rng = jax.random.split(rng)
  if use_conditional_init and dummy_inputs is not None and 'smiles' in dummy_inputs:
    tokens = dummy_inputs['smiles']
    zt = model.apply(
        variables,
        tokens,
        method=model.conditional_sample,
        rngs={'sample': sub_rng}
    )
  else:
    zt = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={'sample': sub_rng},
    )
  rng, sub_rng = jax.random.split(rng)

  def body_fn(i, zt):
    return model.apply(
        variables,
        sub_rng,
        i,
        model.timesteps,
        zt,
        conditioning=conditioning,
        method=model.sample_step,
    )

  z0 = jax.lax.fori_loop(
      lower=0, upper=model.timesteps, body_fun=body_fn, init_val=zt
  )
  return model.apply(
      variables,
      z0,
      conditioning=conditioning,
      method=model.decode,
      rngs={'sample': rng},
  )


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=0)
def reconstruct(model, train_state, rng, t, inputs, conditioning=None):
  """Reconstruct from the latent at t."""
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  variables = {
      'params': get_attr(train_state, 'ema_params'),
      **get_attr(train_state, 'state'),
  }
  f = model.apply(variables, inputs, conditioning, method=model.encode)

  timesteps = model.timesteps
  tn = jnp.ceil(t * timesteps).astype('int32')
  t = tn / timesteps
  rng, sub_rng = jax.random.split(rng)
  zt = model.apply(
      variables, f, t, method=model.forward_sample, rngs={'sample': sub_rng}
  )
  rng, sub_rng = jax.random.split(rng)

  def body_fn(i, zt):
    return model.apply(
        variables,
        sub_rng,
        i,
        timesteps,
        zt,
        conditioning=conditioning,
        method=model.sample_step,
    )

  z0 = jax.lax.fori_loop(
      lower=timesteps - tn, upper=timesteps, body_fun=body_fn, init_val=zt
  )
  return model.apply(
      variables,
      z0,
      conditioning=conditioning,
      method=model.decode,
      rngs={'sample': rng},
  )
