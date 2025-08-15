"""
Enhanced test script to analyze numerical differences between FP32 and BF16 MD4 models
with detailed per-parameter gradient analysis.  (Fixed RNG, FrozenDict handling,
symmetric relative error, fp32 loss readout, and blocking for determinism.)
"""

from collections.abc import Mapping
import jax
import jax.numpy as jnp
import ml_collections
from flax.core import FrozenDict
from md4.models.utils import get_model


def create_test_config():
    config = ml_collections.ConfigDict()
    # Model parameters
    config.model_type = "md4"
    config.data_shape = (128,)
    config.vocab_size = 4096
    config.timesteps = 1000
    config.noise_schedule = "cosine"
    config.outside_embed = True
    config.time_features = "t"
    config.cont_time = True
    config.fp_bits = 4096
    config.fingerprint_dim = 4096
    config.fingerprint_mlp_layers = (2048, 1024, 512, 256)
    config.feature_dim = 256
    config.n_layers = 8
    config.ch_mult = (1,)
    config.n_dit_layers = 0
    config.dit_num_heads = 12
    config.dit_hidden_size = 768
    config.dropout_rate = 0.0
    config.multiple_of = 256
    config.num_heads = 8
    config.n_kv_heads = 4
    config.mlp_type = "swiglu"
    config.depth_scaled_init = True
    config.cond_type = "adaln_zero"
    config.classes = -1
    config.sampler = "topp"
    config.sampling_grid = "uniform"
    config.topp = 0.98
    # dtype will be set per-config in main()
    config.dtype = jnp.float32
    config.param_dtype = jnp.float32
    return config


def initialize_models_same_weights(config_fp32, config_bf16, rng_key, batch_size):
    """Initialize both models with identical weights & identical dummy input."""
    model_fp32 = get_model(config_fp32)
    model_bf16 = get_model(config_bf16)

    dummy_input = jax.random.randint(
        rng_key, (batch_size, config_fp32.data_shape[0]), 0, config_fp32.vocab_size
    )

    init_key = jax.random.PRNGKey(42)  # fixed for reproducibility
    variables_fp32 = model_fp32.init({"params": init_key, "sample": init_key}, dummy_input)

    # Use the same weights for the bf16 model; only cast params if param_dtype differs.
    variables_bf16 = variables_fp32
    if config_fp32.param_dtype != config_bf16.param_dtype:
        def cast_fn(x):
            return x.astype(config_bf16.param_dtype) if isinstance(x, jnp.ndarray) else x
        variables_bf16 = variables_fp32.copy({
            "params": jax.tree_util.tree_map(cast_fn, variables_fp32["params"])
        })

    return model_fp32, model_bf16, variables_fp32, variables_bf16, dummy_input


def run_forward_backward(model, variables, input_data, rng_key):
    """Run forward and backward pass with a fixed RNG (no stochastic mismatch)."""
    def loss_fn(params):
        # Handle both dict and FrozenDict properly
        if hasattr(variables, 'copy') and hasattr(variables.copy, '__code__'):
            # This is a FrozenDict with copy method that takes arguments
            vars_with_params = variables.copy({"params": params})
        else:
            # This is a regular dict, create a new dict
            vars_with_params = dict(variables)
            vars_with_params["params"] = params
        
        stats = model.apply(vars_with_params, input_data, rngs={"sample": rng_key})
        # Ensure reported scalar is fp32 for fair printing (does not change gradients here)
        loss = stats["loss"].astype(jnp.float32)
        return loss, stats

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, stats), grads = grad_fn(variables["params"])
    # Force computation now for deterministic logging
    loss = jax.block_until_ready(loss)
    grads = jax.tree_util.tree_map(lambda x: x, grads)  # touch all leaves
    return loss, stats, grads


def _is_mapping(x):
    return isinstance(x, (dict, FrozenDict, Mapping))


def analyze_gradient_tree(grads_fp32, grads_bf16, prefix="", results=None):
    """Recursively analyze gradient differences for each parameter (handles FrozenDict)."""
    if results is None:
        results = []

    if _is_mapping(grads_fp32) and _is_mapping(grads_bf16):
        for key in grads_fp32.keys():
            if key in grads_bf16:
                new_prefix = f"{prefix}.{key}" if prefix else key
                analyze_gradient_tree(grads_fp32[key], grads_bf16[key], new_prefix, results)
        return results

    # Leaf handling
    if grads_fp32 is None or grads_bf16 is None:
        return results
    if not isinstance(grads_fp32, jnp.ndarray) or not isinstance(grads_bf16, jnp.ndarray):
        return results

    fp32_grad = grads_fp32.astype(jnp.float32)
    bf16_grad = grads_bf16.astype(jnp.float32)

    abs_diff = jnp.abs(fp32_grad - bf16_grad)
    # symmetric relative error avoids division blowups near zero
    eps = jnp.array(1e-8, dtype=jnp.float32)
    rel_diff = abs_diff / (jnp.abs(fp32_grad) + jnp.abs(bf16_grad) + eps)

    has_nan_fp32 = jnp.any(jnp.isnan(fp32_grad))
    has_inf_fp32 = jnp.any(jnp.isinf(fp32_grad))
    has_nan_bf16 = jnp.any(jnp.isnan(bf16_grad))
    has_inf_bf16 = jnp.any(jnp.isinf(bf16_grad))

    num_params = int(fp32_grad.size)
    stats = {
        "parameter": prefix,
        "shape": tuple(fp32_grad.shape),
        "num_params": num_params,
        "max_abs_diff": float(jnp.max(abs_diff)),
        "mean_abs_diff": float(jnp.mean(abs_diff)),
        "std_abs_diff": float(jnp.std(abs_diff)),
        "max_rel_diff": float(jnp.max(rel_diff)),
        "mean_rel_diff": float(jnp.mean(rel_diff)),
        "std_rel_diff": float(jnp.std(rel_diff)),
        "fp32_max_grad": float(jnp.max(jnp.abs(fp32_grad))),
        "fp32_mean_grad": float(jnp.mean(jnp.abs(fp32_grad))),
        "fp32_std_grad": float(jnp.std(fp32_grad)),
        "bf16_max_grad": float(jnp.max(jnp.abs(bf16_grad))),
        "bf16_mean_grad": float(jnp.mean(jnp.abs(bf16_grad))),
        "bf16_std_grad": float(jnp.std(bf16_grad)),
        "has_nan_fp32": bool(has_nan_fp32),
        "has_inf_fp32": bool(has_inf_fp32),
        "has_nan_bf16": bool(has_nan_bf16),
        "has_inf_bf16": bool(has_inf_bf16),
        "dtype_fp32": str(grads_fp32.dtype),
        "dtype_bf16": str(grads_bf16.dtype),
    }
    results.append(stats)
    return results


def print_detailed_analysis(gradient_stats, loss_fp32, loss_bf16, total_params):
    print("=== DETAILED GRADIENT ANALYSIS ===")
    print(f"Analyzed {len(gradient_stats)} parameter groups with {total_params:,} total parameters\n")

    by_rel = sorted(gradient_stats, key=lambda x: x["max_rel_diff"], reverse=True)
    by_abs = sorted(gradient_stats, key=lambda x: x["max_abs_diff"], reverse=True)
    by_size = sorted(gradient_stats, key=lambda x: x["num_params"], reverse=True)

    print("Top 15 parameters by RELATIVE gradient difference:")
    print("-" * 140)
    print(f"{'Parameter':<50} {'Shape':<15} {'#Params':<8} {'Max Rel':<10} {'Mean Rel':<10} "
          f"{'FP32 Max':<10} {'BF16 Max':<10} {'Issues':<15}")
    print("-" * 140)
    for s in by_rel[:15]:
        issues = []
        if s["has_nan_fp32"] or s["has_inf_fp32"]:
            issues.append("FP32:NaN/Inf")
        if s["has_nan_bf16"] or s["has_inf_bf16"]:
            issues.append("BF16:NaN/Inf")
        if s["max_rel_diff"] > 1.0:
            issues.append("LargeRelDiff")
        print(f"{s['parameter']:<50} {str(s['shape']):<15} {s['num_params']:<8} "
              f"{s['max_rel_diff']:<10.4f} {s['mean_rel_diff']:<10.4f} "
              f"{s['fp32_max_grad']:<10.4f} {s['bf16_max_grad']:<10.4f} {(','.join(issues) or 'None'):<15}")

    print("\nTop 10 parameters by ABSOLUTE gradient difference:")
    print("-" * 120)
    print(f"{'Parameter':<50} {'Shape':<15} {'Max Abs Diff':<12} {'Mean Abs Diff':<13} {'FP32 Std':<10} {'BF16 Std':<10}")
    print("-" * 120)
    for s in by_abs[:10]:
        print(f"{s['parameter']:<50} {str(s['shape']):<15} {s['max_abs_diff']:<12.6f} "
              f"{s['mean_abs_diff']:<13.6f} {s['fp32_std_grad']:<10.4f} {s['bf16_std_grad']:<10.4f}")

    print("\nLargest parameter groups:")
    print("-" * 100)
    print(f"{'Parameter':<50} {'Shape':<15} {'#Params':<10} {'Max Rel Diff':<12} {'Max Abs Diff':<12}")
    print("-" * 100)
    for s in by_size[:10]:
        print(f"{s['parameter']:<50} {str(s['shape']):<15} {s['num_params']:<10} "
              f"{s['max_rel_diff']:<12.6f} {s['max_abs_diff']:<12.6f}")

    # Overall
    total_max_rel = max(s["max_rel_diff"] for s in gradient_stats) if gradient_stats else 0.0
    w_mean_rel = (sum(s["mean_rel_diff"] * s["num_params"] for s in gradient_stats) / total_params) if total_params else 0.0
    total_max_abs = max(s["max_abs_diff"] for s in gradient_stats) if gradient_stats else 0.0
    w_mean_abs = (sum(s["mean_abs_diff"] * s["num_params"] for s in gradient_stats) / total_params) if total_params else 0.0

    print("\n=== OVERALL STATISTICS ===")
    print(f"Loss difference (absolute): {float(jnp.abs(loss_fp32 - loss_bf16)):.8f}")
    denom = float(jnp.abs(loss_fp32)) + 1e-10
    print(f"Loss difference (relative): {float(jnp.abs(loss_fp32 - loss_bf16) / denom):.8f}")
    print(f"Max relative gradient difference: {total_max_rel:.6f}")
    print(f"Weighted mean relative gradient difference: {w_mean_rel:.6f}")
    print(f"Max absolute gradient difference: {total_max_abs:.6f}")
    print(f"Weighted mean absolute gradient difference: {w_mean_abs:.6f}")

    severe = [s for s in gradient_stats if s["max_rel_diff"] > 0.5]
    moderate = [s for s in gradient_stats if 0.1 < s["max_rel_diff"] <= 0.5]
    mild = [s for s in gradient_stats if 0.01 < s["max_rel_diff"] <= 0.1]
    good = [s for s in gradient_stats if s["max_rel_diff"] <= 0.01]

    print("\n=== PARAMETER CATEGORIZATION BY GRADIENT DIFFERENCE ===")
    print(f"Severe (>50% rel diff):     {len(severe):3d} parameter groups ({sum(s['num_params'] for s in severe):,} params)")
    print(f"Moderate (10-50% rel diff): {len(moderate):3d} parameter groups ({sum(s['num_params'] for s in moderate):,} params)")
    print(f"Mild (1-10% rel diff):      {len(mild):3d} parameter groups ({sum(s['num_params'] for s in mild):,} params)")
    print(f"Good (<1% rel diff):        {len(good):3d} parameter groups ({sum(s['num_params'] for s in good):,} params)")

    if severe:
        print("\nSEVERE ISSUES DETECTED:")
        for s in severe:
            print(f"  {s['parameter']}: {s['max_rel_diff']:.3f} max rel diff, {s['num_params']} params")

    nan_inf = [s for s in gradient_stats if s["has_nan_fp32"] or s["has_inf_fp32"] or s["has_nan_bf16"] or s["has_inf_bf16"]]
    if nan_inf:
        print("\nNaN/Inf ISSUES DETECTED:")
        for s in nan_inf:
            issues = []
            if s["has_nan_fp32"]: issues.append("FP32 NaN")
            if s["has_inf_fp32"]: issues.append("FP32 Inf")
            if s["has_nan_bf16"]: issues.append("BF16 NaN")
            if s["has_inf_bf16"]: issues.append("BF16 Inf")
            print(f"  {s['parameter']}: {', '.join(issues)}")

    print("\n=== RECOMMENDATIONS ===")
    if total_max_rel > 0.1:
        print("⚠️  WARNING: Large gradient differences detected!")
        print("Recommendations:")
        print("  1) Use identical RNGs across dtypes (done here).")
        print("  2) Compute CE/loss in fp32 inside the model if possible.")
        print("  3) Keep norms, QK^T/AV matmuls, gated-MLP product, and final head in fp32 compute.")
        print("  4) Consider jax.default_matmul_precision('highest') for both runs for apples-to-apples.")
    else:
        print("✅ Gradient differences are within acceptable range for mixed precision training.")


def main():
    print("Enhanced MD4 Mixed Precision Numerical Analysis")
    print("=" * 70)

    # Optionally standardize matmul precision for both runs
    # with jax.default_matmul_precision('highest'):
    config_fp32 = create_test_config()
    config_fp32.dtype = jnp.float32
    config_fp32.param_dtype = jnp.float32

    config_bf16 = create_test_config()
    config_bf16.dtype = jnp.bfloat16
    config_bf16.param_dtype = jnp.float32  # fp32 weights

    print(f"FP32 Config: dtype={config_fp32.dtype}, param_dtype={config_fp32.param_dtype}")
    print(f"BF16 Config: dtype={config_bf16.dtype}, param_dtype={config_bf16.param_dtype}\n")

    rng_key = jax.random.PRNGKey(42)
    batch_size = 4

    print("Initializing models with identical weights...")
    model_fp32, model_bf16, vars_fp32, vars_bf16, test_input = initialize_models_same_weights(
        config_fp32, config_bf16, rng_key, batch_size
    )

    # Count total parameters
    leaves = [x for x in jax.tree_util.tree_leaves(vars_fp32["params"]) if isinstance(x, jnp.ndarray)]
    total_params = sum(int(x.size) for x in leaves)
    print(f"Total parameters: {total_params:,}\n")

    print("Running forward and backward passes...")
    # Use the SAME rng for both passes to avoid stochastic differences
    sample_rng = jax.random.PRNGKey(123)

    loss_fp32, stats_fp32, grads_fp32 = run_forward_backward(model_fp32, vars_fp32, test_input, sample_rng)
    loss_bf16, stats_bf16, grads_bf16 = run_forward_backward(model_bf16, vars_bf16, test_input, sample_rng)

    print(f"FP32 Loss: {float(loss_fp32):.8f}")
    print(f"BF16 Loss: {float(loss_bf16):.8f}\n")

    print("Analyzing gradients...")
    gradient_stats = analyze_gradient_tree(grads_fp32, grads_bf16)

    print_detailed_analysis(gradient_stats, loss_fp32, loss_bf16, total_params)


if __name__ == "__main__":
    main()
