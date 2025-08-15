"""Network adapters and utility modules."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class FingerprintAdapter(nn.Module):
    """
    A module that adapts a raw fingerprint to a desired dimension.

    Attributes:
        raw_fingerprint_dim: The dimension of the raw fingerprint.
        fingerprint_dim: The desired dimension of the fingerprint.
        layers: Number of layers in the adapter.
        dtype: The dtype of the computation (default: jnp.float32).
        param_dtype: The dtype of the parameters (default: jnp.float32).
    """
    raw_fingerprint_dim: int = 4096
    fingerprint_dim: int = 2048
    layers: int = 2
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x: The input data (raw fingerprint).

        Returns:
            The adapted fingerprint.
        """

        if self.raw_fingerprint_dim != self.fingerprint_dim:
            x = jnp.where(
                x < 0.5,
                0,
                1
            )
            x = jnp.logical_or(
                x[:, :self.fingerprint_dim],
                x[:, self.fingerprint_dim:],
            )
 
        for i in range(self.layers - 1):
            x = nn.Dense(
                features=self.raw_fingerprint_dim,
                name=f"fingerprint_adapter_dense_{i}",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
            x = nn.relu(x)

        x = nn.Dense(
            features=self.fingerprint_dim,
            name="fingerprint_adapter_out",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        logits = x

        output = jnp.where(
            nn.sigmoid(logits) < 0.5,
            0.0,
            1.0
        )

        return output.astype(self.dtype), logits  # Ensure output is consistent with dtype


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP).

    Attributes:
        features: A sequence of integers, where each integer is the number of
                  neurons in a layer. The length of the sequence determines
                  the number of layers.
        dtype: The dtype of the computation (default: jnp.float32).
        param_dtype: The dtype of the parameters (default: jnp.float32).
    """
    features: Sequence[int]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The output of the model.
        """
        # The first layers will have a Swish activation function.
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(
                features=feat,
                name=f"cond_dense_{i}",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
            x = nn.swish(x)
        
        # The final layer is the output layer and typically doesn't have an
        # activation function applied here (it might be applied in the loss function).
        x = nn.Dense(
            features=self.features[-1],
            name="cond_dense_out",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return x
