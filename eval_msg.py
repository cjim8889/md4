#!/usr/bin/env python3
"""Evaluation script for MSG dataset: load checkpoints and generate SMILES."""

import dataclasses
import os
from typing import Any, Dict, List

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import polars as pl
import transformers
from absl import app, flags, logging
from orbax import checkpoint as orbax_checkpoint
from tqdm import tqdm

# Import MD4 modules
from md4 import sampling, train, utils
from md4.configs.md4 import molecular

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "", "Directory containing checkpoints")
flags.DEFINE_string("eval_data", "./data/msg/msg_processed.parquet", "Path to MSG eval dataset")
flags.DEFINE_string("output_file", "./results/msg_eval_results.csv", "Output file for results")
flags.DEFINE_string("intermediate_dir", "./results/intermediate", "Directory for intermediate batch results")
flags.DEFINE_integer("num_samples", 10, "Number of SMILES to generate per data point")
flags.DEFINE_integer("checkpoint_step", -1, "Checkpoint step to load (-1 for latest)")
flags.DEFINE_string("tokenizer_path", "data/smiles_tokenizer", "Path to SMILES tokenizer")
flags.DEFINE_integer("batch_size", 32, "Batch size for generation")


@dataclasses.dataclass
class EvalResults:
    """Container for evaluation results."""
    original_smiles: List[str]
    original_fingerprints: List[np.ndarray]
    generated_smiles: List[List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "original_smiles": self.original_smiles,
            "original_fingerprints": [fp.tolist() for fp in self.original_fingerprints],
            "generated_smiles": self.generated_smiles,
        }


def load_config() -> ml_collections.ConfigDict:
    """Load molecular configuration."""
    config = molecular.get_config()
    # Override some settings for evaluation
    config.batch_size = FLAGS.batch_size
    return config


def load_checkpoint_manager(checkpoint_dir: str) -> orbax_checkpoint.CheckpointManager:
    """Load checkpoint manager."""
    checkpointers = dict(
        train_state=orbax_checkpoint.PyTreeCheckpointer(),
    )
    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointers=checkpointers,
        options=orbax_checkpoint.CheckpointManagerOptions(create=False),
    )


def load_model_and_state(config: ml_collections.ConfigDict, checkpoint_dir: str) -> tuple[nn.Module, train.TrainState]:
    """Load model and checkpoint state."""
    # Initialize model
    rng = utils.get_rng(config.seed)
    data_shape = (config.max_length,)
    
    # Create dummy schedule function (not used for inference)
    schedule_fn = lambda step: config.learning_rate
    
    # Create model and train state
    model, optimizer, train_state, metrics_class = train.create_train_state(
        config,
        rng,
        input_shape=(config.batch_size,) + data_shape,
        schedule_fn=schedule_fn,
    )
    
    # Load checkpoint
    checkpoint_manager = load_checkpoint_manager(checkpoint_dir)
    if FLAGS.checkpoint_step >= 0:
        step = FLAGS.checkpoint_step
    else:
        step = checkpoint_manager.latest_step()
    
    if step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    logging.info(f"Loading checkpoint from step {step}")
    
    checkpointed_state = {"train_state": train_state}
    checkpointed_state = checkpoint_manager.restore(step, items=checkpointed_state)
    train_state = checkpointed_state["train_state"]
    
    # No need to replicate for evaluation
    return model, train_state


def load_eval_data(eval_data_path: str) -> pl.DataFrame:
    """Load MSG evaluation dataset."""
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data not found: {eval_data_path}")
    
    df = pl.read_parquet(eval_data_path)
    logging.info(f"Loaded {len(df)} evaluation samples")
    return df


def load_tokenizer(tokenizer_path: str) -> transformers.PreTrainedTokenizerFast:
    """Load SMILES tokenizer."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        logging.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def generate_samples_for_batch(
    model: nn.Module,
    train_state: train.TrainState,
    fingerprints: np.ndarray,
    tokenizer: transformers.PreTrainedTokenizerFast,
    config: ml_collections.ConfigDict,
    num_samples: int,
    rng: jnp.ndarray,
) -> List[List[str]]:
    """Generate multiple SMILES samples for a batch of fingerprints."""
    batch_size = fingerprints.shape[0]
    
    # Repeat fingerprints num_samples times
    # Shape: (batch_size * num_samples, fingerprint_dim)
    conditioning = jnp.repeat(jnp.array(fingerprints, dtype=jnp.int32), num_samples, axis=0)
    
    # Generate all samples in one call
    samples = sampling.simple_generate(
        rng,
        train_state,
        batch_size * num_samples,
        model,
        conditioning=conditioning,
    )
    
    # Reshape samples to (batch_size, num_samples, sequence_length)
    samples = samples.reshape(batch_size, num_samples, -1)
    
    # Convert to list of lists of SMILES strings
    batch_results = []
    for batch_idx in range(batch_size):
        sample_list = []
        for sample_idx in range(num_samples):
            tokens = samples[batch_idx, sample_idx]
            # Detokenize
            smiles = tokenizer.decode(tokens, skip_special_tokens=True)
            sample_list.append(smiles)
        batch_results.append(sample_list)
    
    return batch_results


def save_batch_results(
    batch_idx: int,
    batch_smiles: List[str],
    batch_fingerprints: np.ndarray,
    batch_generated: List[List[str]],
    intermediate_dir: str,
):
    """Save results for a single batch to CSV."""
    import json
    
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Convert batch to rows
    batch_data = []
    batch_size = len(batch_smiles)

    original_smiles_list = []
    generated_smiles_list = []
    
    for i in range(batch_size):
        for j, gen_smi in enumerate(batch_generated[i]):
            original_smiles_list.append(batch_smiles[i])
            generated_smiles_list.append(gen_smi)

    batch_data = {
        "original_smiles": original_smiles_list,
        "generated_smiles": generated_smiles_list,
    }

    # Save to CSV
    batch_df = pl.DataFrame(batch_data)
    batch_file = os.path.join(intermediate_dir, f"batch_{batch_idx:04d}.csv")
    batch_df.write_csv(batch_file)
    
    logging.info(f"Saved batch {batch_idx} results to {batch_file}")


def run_evaluation(
    model: nn.Module,
    train_state: train.TrainState,
    eval_df: pl.DataFrame,
    tokenizer: transformers.PreTrainedTokenizerFast,
    config: ml_collections.ConfigDict,
    num_samples: int,
    intermediate_dir: str,
) -> EvalResults:
    """Run evaluation on the dataset."""
    rng = utils.get_rng(42)  # Fixed seed for reproducible evaluation
    
    results = EvalResults(
        original_smiles=[],
        original_fingerprints=[],
        generated_smiles=[],
    )
    
    # Process in batches, drop incomplete last batch
    batch_size = config.batch_size
    num_complete_batches = len(eval_df) // batch_size
    num_samples_processed = num_complete_batches * batch_size
    
    logging.info(f"Processing {num_samples_processed}/{len(eval_df)} samples in {num_complete_batches} complete batches")
    logging.info(f"Intermediate batch results will be saved to: {intermediate_dir}")
    if num_samples_processed < len(eval_df):
        logging.info(f"Dropping {len(eval_df) - num_samples_processed} samples from incomplete last batch")
    
    for batch_idx in tqdm(range(num_complete_batches), desc="Generating samples"):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_df = eval_df[start_idx:end_idx]
        
        # Extract data
        batch_smiles = batch_df["smiles"].to_list()
        batch_fingerprints = np.array(batch_df["fingerprint"].to_list())
        
        rng, batch_rng = jax.random.split(rng)
        
        # Generate samples
        batch_generated = generate_samples_for_batch(
            model,
            train_state,
            batch_fingerprints,
            tokenizer,
            config,
            num_samples,
            batch_rng,
        )
        
        # Save batch results to intermediate CSV
        save_batch_results(
            batch_idx,
            batch_smiles,
            batch_fingerprints,
            batch_generated,
            intermediate_dir,
        )
        
        # Collect results
        for i in range(batch_size):
            results.original_smiles.append(batch_smiles[i])
            results.original_fingerprints.append(batch_fingerprints[i])
            results.generated_smiles.append(batch_generated[i])
    
    logging.info(f"Generated {len(results.original_smiles)} sets of samples")
    return results


def combine_intermediate_results(intermediate_dir: str, output_file: str):
    """Combine all intermediate batch CSV files into final output."""
    if not os.path.exists(intermediate_dir):
        logging.warning(f"Intermediate directory {intermediate_dir} not found, skipping combination")
        return
    
    # Find all batch CSV files
    batch_files = sorted([
        f for f in os.listdir(intermediate_dir) 
        if f.startswith("batch_") and f.endswith(".csv")
    ])
    
    if not batch_files:
        logging.warning(f"No batch files found in {intermediate_dir}")
        return
    
    logging.info(f"Combining {len(batch_files)} batch files into {output_file}")
    
    # Read and combine all batch files
    all_dfs = []
    for batch_file in batch_files:
        batch_path = os.path.join(intermediate_dir, batch_file)
        batch_df = pl.read_csv(batch_path)
        all_dfs.append(batch_df)
    
    # Combine all DataFrames
    combined_df = pl.concat(all_dfs)
    
    # Save as parquet (more efficient for large files)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.write_csv(output_file)
    
    logging.info(f"Combined results saved to {output_file}")
    logging.info(f"Total samples: {len(combined_df)}")


def save_results(results: EvalResults, output_file: str):
    """Save evaluation results."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to DataFrame
    data = []
    for i, (orig_smiles, orig_fp, gen_smiles) in enumerate(
        zip(results.original_smiles, results.original_fingerprints, results.generated_smiles)
    ):
        for j, gen_smi in enumerate(gen_smiles):
            data.append({
                "sample_id": i,
                "original_smiles": orig_smiles,
                "generated_smiles": gen_smi,
                "generation_idx": j,
                "original_fingerprint": orig_fp.tolist(),
            })
    
    df = pl.DataFrame(data)
    df.write_parquet(output_file)
    logging.info(f"Results saved to {output_file}")
    
    # Print summary
    logging.info(f"Evaluation summary:")
    logging.info(f"- Total original molecules: {len(results.original_smiles)}")
    logging.info(f"- Samples per molecule: {FLAGS.num_samples}")
    logging.info(f"- Total generated samples: {len(data)}")


def main(argv):
    del argv  # Unused
    
    # Validate inputs
    if not FLAGS.checkpoint_dir:
        raise ValueError("Must specify --checkpoint_dir")
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {FLAGS.checkpoint_dir}")
    
    logging.info("Starting MSG evaluation...")
    
    # Load configuration
    config = load_config()
    logging.info(f"Loaded config with vocab_size={config.vocab_size}")
    
    # Load model and checkpoint
    model, train_state = load_model_and_state(config, FLAGS.checkpoint_dir)
    logging.info("Model and checkpoint loaded successfully")
    
    # Load tokenizer
    tokenizer = load_tokenizer(FLAGS.tokenizer_path)
    
    # Load evaluation data
    eval_df = load_eval_data(FLAGS.eval_data)
    
    # Run evaluation
    results = run_evaluation(
        model, train_state, eval_df, tokenizer, config, FLAGS.num_samples, FLAGS.intermediate_dir
    )
    
    # Combine intermediate results into final output
    combine_intermediate_results(FLAGS.intermediate_dir, FLAGS.output_file)
    
    # Also save using the traditional method as backup
    save_results(results, FLAGS.output_file.replace('.parquet', '_backup.parquet'))
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    app.run(main) 