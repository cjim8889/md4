# single_script_test.py

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# For TPU-specific compiler parameters, though the code is general.
# If you are not on TPU, this part of the configuration is ignored.
# try:
#     from jax.experimental.pallas import tpu as pltpu
#     mosaic_params = dict(mosaic_params=dict(
#         dimension_semantics=('parallel', 'parallel')
#     ))
# except ImportError:
    # Fallback for GPU/CPU where mosaic is not used
mosaic_params = {}


# ==============================================================================
# 1. Naive Implementation (for Verification)
# ==============================================================================

def sparse_linear_naive(x, W, b, num_valid_atoms):
  """
  Applies a linear layer to a sparse dimension using masking.
  This is our ground truth for correctness, not for performance.

  Args:
    x: Input tensor of shape (batch, num_atoms, in_dim).
    W: Weight matrix of shape (in_dim, out_dim).
    b: Bias vector of shape (out_dim,).
    num_valid_atoms: Integer tensor of shape (batch,) indicating how many
                     atoms are valid for each batch item.
  
  Returns:
    Output tensor of shape (batch, num_atoms, out_dim) with padding zeroed out.
  """
  # Create a mask of shape (batch, num_atoms)
  # Example: if num_valid_atoms[i] is 5, mask[i] will be [T, T, T, T, T, F, F, ...]
  mask = jnp.arange(x.shape[1]) < num_valid_atoms[:, None]
  
  # Apply the linear transformation to all elements
  y = jnp.einsum('bne,eo->bno', x, W) + b
  
  # Zero out the padded entries using the mask.
  # The mask needs to be expanded to match the output shape for broadcasting.
  return jnp.where(mask[..., None], y, 0)


# ==============================================================================
# 2. Pallas Kernel Implementation
# ==============================================================================

def sparse_linear_pallas_kernel(x_ref, W_ref, b_ref, num_valid_atoms_ref, y_ref):
  """
  Pallas kernel to apply a linear layer on a sparse dimension.
  This code runs on the accelerator (TPU/GPU).
  """
  # Each kernel instance handles one atom for one batch item.
  # program_id(0) is the batch index, program_id(1) is the atom index.
  batch_idx, atom_idx = pl.program_id(0), pl.program_id(1)
  
  # Load the number of valid atoms for the current batch item. This is a scalar load.
  # num_valid_atoms_ref has block shape (1,), so index locally
  n_valid = num_valid_atoms_ref[0]
  
  # Always initialize the output block to zeros (handles padded atoms deterministically).
  y_ref[0, 0, :] = jnp.zeros_like(y_ref[0, 0, :])

  # --- The Core Optimization ---
  # Only perform computation if the current atom is not padding.
  # Use pl.when as a callable guard around the compute block.
  def _compute():
    # x_ref has block shape (1, 1, in_dim); index locally within the block
    x_vec = x_ref[0, 0, :]

    # Load weights and bias. Pallas may cache these since they are reused.
    # Shapes: (in_dim, out_dim) and (out_dim,)
    W = W_ref[:, :]
    b = b_ref[:]

    # Perform the linear transformation and write to the output slot
    # (in_dim,) @ (in_dim, out_dim) -> (out_dim,)
    y_ref[0, 0, :] = (jnp.dot(x_vec, W) + b).astype(y_ref.dtype)

  pl.when(atom_idx < n_valid)(_compute)

def sparse_linear_pallas(x, W, b, num_valid_atoms):
  """
  User-facing function that configures and invokes the Pallas kernel.
  """
  # Get shapes for configuration
  batch_size, n_atoms, in_dim = x.shape
  out_dim = W.shape[1]
  
  # Pre-initialize the output array with zeros. The kernel will only
  # write to the valid (non-padded) locations.
  y_initial = jnp.zeros((batch_size, n_atoms, out_dim), dtype=x.dtype)
  
  # Invoke the kernel using pallas_call
  # Use interpret mode on CPU backend where only interpret is supported.
  is_cpu = (jax.default_backend() == 'cpu') or all(d.platform == 'cpu' for d in jax.devices())
  y = pl.pallas_call(
    sparse_linear_pallas_kernel,
    # Launch one program per (batch, atom)
    grid=(batch_size, n_atoms),
    # Map arguments to refs using BlockSpec(block_shape, index_map)
    in_specs=[
      # Provide a (1,1,in_dim) block for each (batch, atom)
      pl.BlockSpec((1, 1, in_dim), lambda i, j: (i, j, 0)),
      # Read-only full-array refs for W and b
      pl.BlockSpec(None, None),
      pl.BlockSpec(None, None),
      # Provide a (1,) block corresponding to the batch entry's n_valid
      pl.BlockSpec((1,), lambda i, j: (i,)),
    ],
    # Each program writes a (1,1,out_dim) block at (batch, atom)
    out_specs=pl.BlockSpec((1, 1, out_dim), lambda i, j: (i, j, 0)),
  # Describe the full output shape/dtype
  out_shape=jax.ShapeDtypeStruct(y_initial.shape, y_initial.dtype),
  # Compiler hints for performance (e.g., for TPUs)
  compiler_params=mosaic_params,
  interpret=is_cpu,
  name="sparse_linear_pallas",
  )(x, W, b, num_valid_atoms)
  
  return y


# ==============================================================================
# 3. Verification and Usage Example
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 256
    NUM_ATOMS = 128
    IN_DIM = 32
    OUT_DIM = 64
    
    # Use a key for reproducible randomness
    key = jax.random.PRNGKey(0)
    key_x, key_w, key_b, key_n = jax.random.split(key, 4)
    
    # --- Problem Setup ---
    # Input tensor
    x = jax.random.normal(key_x, (BATCH_SIZE, NUM_ATOMS, IN_DIM))
    
    # Linear layer weights and bias
    W = jax.random.normal(key_w, (IN_DIM, OUT_DIM))
    b = jax.random.normal(key_b, (OUT_DIM,))
    
    # The crucial sparsity information: an array of integers
    # where each int is the number of valid atoms for that batch item.
    # We generate random lengths between a min value and the max.
    # In a real scenario, this would come from your data preprocessing.
    min_valid = 50 
    num_valid_atoms = jax.random.randint(
        key_n, (BATCH_SIZE,), min_valid, NUM_ATOMS + 1
    )
    
    print(f"Shapes:")
    print(f"  Input x: {x.shape}")
    print(f"  Weights W: {W.shape}")
    print(f"  Bias b: {b.shape}")
    print(f"  num_valid_atoms: {num_valid_atoms.shape}")
    print(f"Example `num_valid_atoms`: {num_valid_atoms[:5]}...")

  # --- Run and Compare ---
    
    # JIT-compile both functions for a fair performance comparison
    jitted_naive = jax.jit(sparse_linear_naive)
    # On CPU, Pallas only supports interpret mode; avoid JIT wrapping to prevent lowering error.
    is_cpu = (jax.default_backend() == 'cpu') or all(d.platform == 'cpu' for d in jax.devices())
    jitted_pallas = sparse_linear_pallas if is_cpu else jax.jit(sparse_linear_pallas)
      
    print("\nRunning Naive Implementation (for correctness check)...")
    y_naive = jitted_naive(x, W, b, num_valid_atoms).block_until_ready()
      
    print("Running Pallas Implementation...")
    y_pallas = jitted_pallas(x, W, b, num_valid_atoms).block_until_ready()
      
    # Verify that the results are identical
    are_close = jnp.allclose(y_naive, y_pallas, atol=1e-5, rtol=1e-5)
      
    print(f"\nVerification successful: {are_close}")
    assert are_close, "Pallas implementation does not match the naive version!"

    # --- Sanity Check Inspection ---
    # You can also inspect the output to be sure.
    # For a batch item with N valid atoms, y[i, N:, :] should be all zeros.
    batch_item_idx = 0
    n_valid_for_item = num_valid_atoms[batch_item_idx].item()
    
    print(f"\nInspecting batch item {batch_item_idx} (valid atoms: {n_valid_for_item}):")
    
    # Check the last valid atom (should be non-zero)
    last_valid_output = y_pallas[batch_item_idx, n_valid_for_item - 1, :]
    print(f"  Output at last valid atom [{n_valid_for_item-1}]: non-zero? "
      f"{not jnp.all(last_valid_output == 0)}")

    # Check the first padded atom (should be zero)
    if n_valid_for_item < NUM_ATOMS:
      first_padded_output = y_pallas[batch_item_idx, n_valid_for_item, :]
      print(f"  Output at first padded atom [{n_valid_for_item}]: all-zero? "
        f"{jnp.all(first_padded_output == 0)}")