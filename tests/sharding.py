import os

# Set this to True to run the model on CPU only.
flags = os.environ.get("XLA_FLAGS", "")
flags += " --xla_force_host_platform_device_count=16"  # Simulate 8 devices
os.environ["XLA_FLAGS"] = flags

import math
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp


import numpy as np

mesh_rows = 2
mesh_cols = jax.device_count() // 2
global_shape = (2880, 128)


mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ("model", "data"))
sharding = jax.sharding.NamedSharding(mesh, P("data", None))
inp_data = jnp.zeros(global_shape)

for d, index in sharding.addressable_devices_indices_map(global_shape).items():
    print(f"device: {d}, index first dimension: {index[0]} ")


arrays = [
    jax.device_put(inp_data[index], d)
    for d, index in sharding.addressable_devices_indices_map(global_shape).items()
]
arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
assert arr.shape == (2880, 128)
