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

"""Utils."""

import math
import time
from collections.abc import Mapping
from typing import Any, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from clu import platform
from orbax import checkpoint as orbax_checkpoint


def loss2bpt(loss_dict, data_shape):
    """Normalize loss to bits per token."""
    seq_len = jnp.prod(jnp.array(data_shape))
    rescale_to_bpd = 1.0 / (seq_len * jnp.log(2.0))
    bpt_loss_dict = {}
    for k, v in loss_dict.items():
        if "loss" in k:
            bpt_loss_dict[k] = v * rescale_to_bpd
        else:
            bpt_loss_dict[k] = v
    return bpt_loss_dict


def constant_init(value, dtype="float32"):
    def _init(key, shape, dtype=dtype):
        del key
        return value * jnp.ones(shape, dtype)

    return _init


def _logistic_pdf_fn(z, log_scales):
    return -z - 2.0 * jax.nn.softplus(-z) - log_scales


class DiscretizedLogisticMixture(distrax.Distribution):
    """Discretized mixture of Logistics defined in PixelCNN++."""

    def __init__(
        self,
        w_logits,
        locs,
        log_scales,
        min_val=0.0,
        max_val=255.0,
        bin_size=1.0,
    ):
        self._w_logits = w_logits
        self._locs = locs
        self._log_scales = log_scales
        self._min_val = min_val
        self._max_val = max_val
        self._bin_size = bin_size

        self._batch_shape = jax.lax.broadcast_shapes(
            self._locs.shape[:-1], self._log_scales.shape[:-1]
        )

        self._cdf = jax.nn.sigmoid
        self._log_prob_fn = _logistic_pdf_fn

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of event of distribution samples."""
        return ()

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    @property
    def _n_components(self) -> int:
        return self._locs.shape[-1]

    def _mean(self):
        # The expected value of the mixture is the sum of the linear probabilities
        # of each component multiplied by their means.
        # Apply softmax on the logits to get the linear probabilities.
        probs = jax.nn.softmax(self._w_logits)
        return jnp.sum(self._locs * probs, axis=-1)

    def log_prob(self, event: chex.Array) -> chex.Array:
        # expand the mixture dim
        event = jnp.expand_dims(event, -1)
        assert len(self._locs.shape) == len(event.shape)

        # Expand the dimensions of the params for tiling and broadcasting.
        locs = self._locs
        w_logits = self._w_logits
        inv_scales = jnp.exp(-self._log_scales)

        # pdf at the mid of the bin, used when bins are too small
        z = (event - locs) * inv_scales
        mid_log_prob = self._log_prob_fn(z, self._log_scales)

        # Calculate difference of sigmoid.
        half_bin = self._bin_size / 2
        b = (event - locs + half_bin) * inv_scales
        a = (event - locs - half_bin) * inv_scales
        a = self._cdf(a)
        b = self._cdf(b)
        diff = b - a

        # Handle edge case.
        edge_b = jax.nn.sigmoid((self._min_val - locs + half_bin) * inv_scales) - 0.0
        edge_a = 1.0 - jax.nn.sigmoid((self._max_val - locs - half_bin) * inv_scales)
        diff = jnp.where(event > self._max_val - half_bin, edge_a, diff)
        diff = jnp.where(event < self._min_val + half_bin, edge_b, diff)

        # Avoid small values for the subsequent log operation.
        diff = jnp.maximum(diff, 1e-12)
        log_prob = jnp.where(diff > 1e-8, jnp.log(diff), mid_log_prob)

        # Normalize logits.
        w_logits -= jax.nn.logsumexp(w_logits, axis=-1, keepdims=True)

        # Total loss, summed over the mixture dimension.
        total = w_logits + log_prob.sum(axis=-2)
        total = jax.nn.logsumexp(total, -1)
        return total

    def _sample_n(self, key: chex.PRNGKey, n: int):
        # First sample from the mixing weights.
        w_dist = distrax.Categorical(logits=self._w_logits)
        key, sub_key = jax.random.split(key)
        index = w_dist.sample(seed=sub_key, sample_shape=n)
        index = jax.nn.one_hot(index, num_classes=self._n_components)

        # Pick the mixture (per pixel) and board cast to n samples.
        log_scales = jnp.sum(jnp.expand_dims(self._log_scales, 0) * index, -1)
        loc = jnp.sum(jnp.expand_dims(self._locs, 0) * index, -1)
        scales = jnp.exp(log_scales)

        # Compute logistic
        _, sub_key = jax.random.split(key)
        logistic_sample = jax.random.logistic(sub_key, shape=loc.shape)
        sample_values = loc + scales * logistic_sample

        return jnp.clip(jnp.round(sample_values), 0, 255).astype("int32")


def shifted_softplus(x, b=1.0):
    """log(exp(x) + b)."""
    return x + jax.nn.softplus(jnp.log(b) - x)


def reverse_broadcast(value, ndim):
    """Broadcast by adding singleton axes to the right, instead of to the left."""
    if value.ndim > ndim:
        raise ValueError(
            f"Cannot reverse broadcast a value with {value.ndim} dimensions "
            f"to {ndim} dimensions."
        )

    if value.ndim < ndim:
        difference = ndim - value.ndim
        return value.reshape(value.shape + difference * (1,))
    else:
        return value


def get_rng(seed: None | int | tuple[int, int]) -> np.ndarray:
    """Returns a JAX RNGKey."""
    if seed is None:
        # Case 1: No random seed given, use XManager ID.
        # All processes (and restarts) get exactly the same seed but every work unit
        # and experiment is different.
        work_unit = platform.work_unit()
        rng = (work_unit.experiment_id, work_unit.id)
    elif isinstance(seed, int):
        # Case 2: Single integer given.
        rng = (0, seed)
    else:
        # Case 3: tuple[int, int] given.
        if not isinstance(seed, (tuple, list)) or len(seed) != 2:
            raise ValueError(
                "Random seed must be an integer or tuple of 2 integers "
                f"but got {seed!r}"
            )
        rng = seed
    # JAX RNGKeys are arrays of np.uint32 and shape [2].
    return np.asarray(rng, dtype=np.uint32)


class StepTraceContextHelper:
    """Helper class to use jax.profiler.StepTraceAnnotation."""

    def __init__(self, name: str, init_step_num: int):
        self.name = name
        self.step_num = init_step_num
        self.context = None

    def __enter__(self):
        self.context = jax.profiler.StepTraceAnnotation(
            self.name, step_num=self.step_num
        )
        self.step_num += 1
        self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        assert self.context is not None, "Exited context without entering."
        self.context.__exit__(exc_type, exc_value, tb)
        self.context = None

    def next_step(self):
        if self.context is None:
            raise ValueError("Must call next_step() within a context.")
        self.__exit__(None, None, None)
        self.__enter__()


def plot_embeddings(step, workdir, embeddings, annotations=None):
    """Helper function to plot embeddings."""
    fig, ax = plt.subplots()
    ax.set_title("Embeddings")
    if embeddings.ndim == 1:
        ax.scatter(np.arange(256), embeddings)
    else:
        assert embeddings.ndim == 2
        colors = np.linspace(0, 1, embeddings.shape[0])
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap="rainbow")
        if annotations:
            for i in range(embeddings.shape[0]):
                ax.annotate(annotations[i], (embeddings[i, 0], embeddings[i, 1]))
    embedding_plot = workdir / "embedding_{}.png".format(step)
    with embedding_plot.open("wb") as f:
        fig.savefig(f)

def generate_image_grids(images):
    """Simple helper to generate a single image from a mini batch."""

    def image_grid(nrow, ncol, imagevecs, imshape):
        images = iter(imagevecs.reshape((-1,) + imshape))
        return jnp.vstack(
            [jnp.hstack([next(images) for _ in range(ncol)][::-1]) for _ in range(nrow)]
        )

    batch_size = images.shape[0]
    grid_size = int(np.floor(np.sqrt(batch_size)))

    image_shape = images.shape[1:]
    return image_grid(
        nrow=grid_size,
        ncol=grid_size,
        imagevecs=images[0 : grid_size**2],
        imshape=image_shape,
    ).astype("uint8")


def detokenize_texts(tokens, tokenizer):
    """Detokenize the outputs."""

    assert len(tokens.shape) == 2, "Invalid token shape."

    np_tokens = np.asarray(tokens)
    detokenized = np.apply_along_axis(tokenizer.decode, -1, np_tokens)

    return detokenized


def get_topk_token_mask(tokenizer, k=100):
    """Get the indices of Top-K tokens."""

    id_unigram_scores = [
        (id_, math.exp(tokenizer._model.GetScore(id_)))  # pylint: disable=protected-access
        for id_ in range(tokenizer.vocab_size)
    ]

    id_unigram_scores_sorted = sorted(id_unigram_scores, key=lambda x: x[1])

    # Exact k elements.
    topk = id_unigram_scores_sorted[-k:]

    topk_mask = [False for _ in range(tokenizer.vocab_size)]
    topk_ids = []

    for id_, _ in topk:
        topk_mask[id_] = True
        topk_ids.append(id_)

    topk_tokens = tokenizer.to_string_list(np.array(topk_ids))

    return np.array(topk_mask), topk_tokens


def reshape_batch(batch: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    """Reshapes a batch to have the leading dimension for the local devices."""
    leading_dims = [jax.local_device_count(), -1]
    return jax.tree_util.tree_map(
        lambda x: np.reshape(x, leading_dims + list(x.shape[1:])), batch
    )


def checkpoints_iterator(ckpt_manager, timeout=None, min_interval_secs=0, period=10000):
    """Repeatedly yield new checkpoints as they appear.

    Args:
      ckpt_manager: CheckpointManager object.
      timeout: int: maximum number of seconds to wait. If left as `None`, then the
        process will wait indefinitely.
      min_interval_secs: int: minimum number of seconds between yielding
        checkpoints.
      period: The period of the checkpoint.

    Yields:
      new checkpoint step.
    """
    last_step = None
    while True:
        cur_step = wait_for_new_checkpoint(
            ckpt_manager, last_step, timeout=timeout, period=period
        )
        if cur_step is None:
            # timed out
            logging.info("Timed-out waiting for a checkpoint.")
            return
        start = time.time()
        last_step = cur_step

        yield cur_step

        time_to_next_eval = start + min_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)


def wait_for_new_checkpoint(
    ckpt_manager: orbax_checkpoint.CheckpointManager,
    last_step=None,
    seconds_to_sleep=1,
    timeout=None,
    period=10000,
):
    """Waits until a new checkpoint file is found.

    Args:
      ckpt_manager: The directory in which checkpoints are saved.
      last_step: The last checkpoint path used or `None` if we're expecting a
        checkpoint for the first time.
      seconds_to_sleep: The number of seconds to sleep for before looking for a
        new checkpoint.
      timeout: The maximum number of seconds to wait. If left as `None`, then the
        process will wait indefinitely.
      period: The period of the checkpoint.

    Returns:
      a new checkpoint path, or None if the timeout was reached.
    """
    logging.info("Waiting for new checkpoint at %s", ckpt_manager.directory)
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
        ckpt_manager.reload()
        cur_step = ckpt_manager.latest_step()
        if cur_step is None or cur_step == last_step or cur_step % period != 0:
            if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
                return None
            time.sleep(seconds_to_sleep)
        else:
            logging.info("Found new checkpoint at step %d", cur_step)
            return cur_step
