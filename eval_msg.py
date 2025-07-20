#!/usr/bin/env python3
"""Evaluation script for MSG dataset: load checkpoints and generate SMILES."""

import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import polars as pl
import transformers
from absl import app, flags, logging
from orbax import checkpoint as orbax_checkpoint
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. InChI to SMILES conversion will not work.")

# Import MD4 modules
from md4 import rdkit_utils, sampling, train, utils
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
    original_inchi: Optional[List[str]] = None  # Store original InChI if available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        result = {
            "original_smiles": self.original_smiles,
            "original_fingerprints": [fp.tolist() for fp in self.original_fingerprints],
            "generated_smiles": self.generated_smiles,
        }
        if self.original_inchi:
            result["original_inchi"] = self.original_inchi
        return result


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
        input_shape=(config.batch_size * 10,) + data_shape,
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
    
    # Replicate train_state for pmap usage across multiple devices
    train_state = flax_utils.replicate(train_state)
    return model, train_state


def load_eval_data(eval_data_path: str) -> pl.DataFrame:
    """Load MSG evaluation dataset."""
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data not found: {eval_data_path}")
    
    df = pl.read_parquet(eval_data_path)
    logging.info(f"Loaded {len(df)} evaluation samples")
    return df


def inchi_to_smiles(inchi: str) -> Optional[str]:
    """Convert InChI to SMILES using RDKit."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for InChI to SMILES conversion")
    
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logging.warning(f"Failed to convert InChI to SMILES: {inchi[:50]}... Error: {e}")
        return None


def detect_molecular_format_and_prepare_data(eval_df: pl.DataFrame, pad_to_length: int = 128) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    """
    Convert InChI data to SMILES and extract atom types and fingerprints. Discard anything that can't be parsed by RDKit.
    
    Args:
        eval_df: DataFrame containing InChI data
        pad_to_length: Length to pad atom types to
    
    Returns:
        Tuple of (smiles_list, atom_types_list, valid_inchi_list, fingerprints_list)
        Only successfully parsed molecules are returned
    """
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required to convert InChI to SMILES. "
            "Please install RDKit: conda install -c conda-forge rdkit"
        )
    
    columns = eval_df.columns
    inchi_columns = [col for col in columns if 'inchi' in col.lower()]
    
    if not inchi_columns:
        raise ValueError("No InChI columns found in the dataframe")
    
    inchi_col = inchi_columns[0]  # Use first InChI column found
    logging.info(f"Found InChI data in column: {inchi_col}")
    
    inchi_list = eval_df[inchi_col].to_list()
    fingerprints_list = eval_df["predicted_fingerprint"].to_list() if "predicted_fingerprint" in eval_df.columns else []
    
    if not fingerprints_list:
        logging.warning("No predicted_fingerprint column found in eval_df, using zero fingerprints")
    
    logging.info("Converting InChI to SMILES and extracting atom types...")
    
    smiles_list = []
    atom_types_list = []
    valid_inchi_list = []
    valid_fingerprints_list = []
    
    for i, inchi in enumerate(tqdm(inchi_list, desc="Converting InChI to SMILES and extracting features")):
        smiles = inchi_to_smiles(inchi)
        if smiles is not None:
            # Extract atom types from converted SMILES
            features = rdkit_utils.process_smiles(smiles, fp_radius=2, fp_bits=2048, pad_to_length=pad_to_length)
            if features is not None and 'atom_types' in features:
                smiles_list.append(smiles)
                atom_types_list.append(features['atom_types'])
                valid_inchi_list.append(inchi)
                
                # Add corresponding fingerprint if available
                if fingerprints_list and i < len(fingerprints_list):
                    valid_fingerprints_list.append(fingerprints_list[i])
                else:
                    # Default to zeros if no fingerprint available
                    valid_fingerprints_list.append(np.zeros(2048))
            # Silently discard molecules that can't be processed
    
    logging.info(f"Successfully processed {len(smiles_list)}/{len(inchi_list)} InChI strings")
    return smiles_list, atom_types_list, valid_inchi_list, valid_fingerprints_list


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
    atom_types: np.ndarray,
    tokenizer: transformers.PreTrainedTokenizerFast,
    config: ml_collections.ConfigDict,
    num_samples: int,
    rng: jnp.ndarray,
) -> List[List[str]]:
    """Generate multiple SMILES samples for a batch of fingerprints."""
    batch_size = fingerprints.shape[0]
    
    # Process fingerprints: convert to binary (0/1) with threshold 0.5 and fold from 4096 to 2048 bits
    processed_fingerprints = process_fingerprints(fingerprints, threshold=0.5, fold_factor=2)
    
    # Repeat fingerprints num_samples times
    # Shape: (batch_size * num_samples, fingerprint_dim)
    expanded_fingerprints = jnp.repeat(jnp.array(processed_fingerprints, dtype=jnp.int32), num_samples, axis=0)
    expanded_atom_types = jnp.repeat(jnp.array(atom_types, dtype=jnp.int32), num_samples, axis=0)

    # Create dummy inputs with the correct shape for the generate function
    total_samples = batch_size * num_samples
    dummy_inputs = jnp.ones((total_samples, config.max_length), dtype="int32")
    
    # Prepare conditioning
    conditioning = {
        "fingerprint": expanded_fingerprints,
        "atom_types": expanded_atom_types,
    }
    
    # Reshape for pmap: need to split across devices
    num_devices = jax.device_count()
    per_device_batch_size = total_samples // num_devices
    
    # Ensure total_samples is divisible by num_devices
    if total_samples % num_devices != 0:
        # Pad to make it divisible
        padding_needed = num_devices - (total_samples % num_devices)
        dummy_inputs = jnp.concatenate([
            dummy_inputs,
            jnp.ones((padding_needed, config.max_length), dtype="int32")
        ], axis=0)
        expanded_fingerprints = jnp.concatenate([
            expanded_fingerprints,
            jnp.zeros((padding_needed, expanded_fingerprints.shape[1]), dtype="int32")
        ], axis=0)
        expanded_atom_types = jnp.concatenate([
            expanded_atom_types,
            jnp.zeros((padding_needed, expanded_atom_types.shape[1]), dtype="int32")
        ], axis=0)
        total_samples_padded = total_samples + padding_needed
        per_device_batch_size = total_samples_padded // num_devices
    else:
        total_samples_padded = total_samples
        padding_needed = 0
    
    # Reshape for pmap: (num_devices, per_device_batch_size, ...)
    dummy_inputs = dummy_inputs.reshape(num_devices, per_device_batch_size, config.max_length)
    conditioning = {
        "fingerprint": expanded_fingerprints.reshape(num_devices, per_device_batch_size, -1),
        "atom_types": expanded_atom_types.reshape(num_devices, per_device_batch_size, -1),
    }
    
    # Replicate the random number generator for pmap
    replicated_rng = flax_utils.replicate(rng)
    
    # Generate all samples using the pmap version
    samples = sampling.generate(
        model,
        train_state,
        replicated_rng,
        dummy_inputs,
        conditioning=conditioning,
    )
    
    # Unreplicate and flatten back
    samples = flax_utils.unreplicate(samples)
    samples = samples.reshape(-1, config.max_length)
    
    # Remove padding if any
    if padding_needed > 0:
        samples = samples[:-padding_needed]
    
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
    batch_inchi: Optional[List[str]] = None,
):
    """Save results for a single batch to CSV."""
    import json
    
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Convert batch to rows
    batch_size = len(batch_smiles)

    original_smiles_list = []
    generated_smiles_list = []
    original_inchi_list = []
    
    for i in range(batch_size):
        for j, gen_smi in enumerate(batch_generated[i]):
            original_smiles_list.append(batch_smiles[i])
            generated_smiles_list.append(gen_smi)
            if batch_inchi is not None:
                original_inchi_list.append(batch_inchi[i])

    batch_data = {
        "original_smiles": original_smiles_list,
        "generated_smiles": generated_smiles_list,
    }
    
    if batch_inchi is not None:
        batch_data["original_inchi"] = original_inchi_list

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
    
    # Convert InChI to SMILES and extract atom types and fingerprints
    smiles_list, atom_types_list, original_inchi_list, fingerprints = detect_molecular_format_and_prepare_data(eval_df)
    
    # Create filtered dataframe with only successfully processed molecules
    eval_df = pl.DataFrame({
        "predicted_fingerprint": fingerprints,
        "atom_types": atom_types_list,
    })
    
    results = EvalResults(
        original_smiles=[],
        original_fingerprints=[],
        generated_smiles=[],
        original_inchi=[],
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
        batch_smiles = smiles_list[start_idx:end_idx]
        batch_fingerprints = np.array(batch_df["predicted_fingerprint"].to_list())
        batch_atom_types = np.array(atom_types_list[start_idx:end_idx])  # Use atom types from our extracted list
        batch_inchi = original_inchi_list[start_idx:end_idx]
        
        rng, batch_rng = jax.random.split(rng)
        
        # Generate samples
        batch_generated = generate_samples_for_batch(
            model,
            train_state,
            batch_fingerprints,
            batch_atom_types,
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
            batch_inchi,
        )
        
        # Collect results
        for i in range(batch_size):
            results.original_smiles.append(batch_smiles[i])
            results.original_fingerprints.append(batch_fingerprints[i])
            results.generated_smiles.append(batch_generated[i])
            results.original_inchi.append(batch_inchi[i])
    
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
            row = {
                "sample_id": i,
                "original_smiles": orig_smiles,
                "generated_smiles": gen_smi,
                "generation_idx": j,
                "original_fingerprint": orig_fp.tolist(),
            }
            
            # Add original InChI if available
            if results.original_inchi is not None:
                row["original_inchi"] = results.original_inchi[i]
            
            data.append(row)
    
    df = pl.DataFrame(data)
    df.write_parquet(output_file)
    logging.info(f"Results saved to {output_file}")
    
    # Print summary
    logging.info(f"Evaluation summary:")
    logging.info(f"- Total original molecules: {len(results.original_smiles)}")
    logging.info(f"- Samples per molecule: {FLAGS.num_samples}")
    logging.info(f"- Total generated samples: {len(data)}")
    if results.original_inchi is not None:
        logging.info(f"- Original format: InChI (converted to SMILES)")
    else:
        logging.info(f"- Original format: SMILES")


def process_fingerprints(fingerprints: np.ndarray, threshold: float = 0.5, fold_factor: int = 2) -> np.ndarray:
    """Process fingerprints by thresholding and folding.
    
    Args:
        fingerprints: Input fingerprints array
        threshold: Threshold for converting to binary (0/1)
        fold_factor: Factor by which to fold the fingerprints
    
    Returns:
        Processed fingerprints array
    """
    if not RDKIT_AVAILABLE:
        logging.warning("RDKit not available, skipping fingerprint processing")
        return fingerprints
    
    try:
        # Convert to binary with threshold
        binary_fingerprints = (fingerprints >= threshold).astype(np.int32)
        
        # Fold the fingerprints using numpy operations
        # Simple folding by taking OR of adjacent bits
        first_half = binary_fingerprints[:, :2048]
        second_half = binary_fingerprints[:, 2048:]
        folded_fingerprints = np.logical_or(first_half, second_half).astype(np.int32)
        return folded_fingerprints
        
    except Exception as e:
        logging.warning(f"Failed to process fingerprints: {e}")
        return fingerprints


def main(argv):
    del argv  # Unused
    
    # # Validate inputs
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
    # save_results(results, FLAGS.output_file.replace('.parquet', '_backup.parquet'))
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    app.run(main)
