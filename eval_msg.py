#!/usr/bin/env python3
"""Evaluation script for MSG dataset: load checkpoints and generate SMILES."""

import argparse
import dataclasses
import multiprocessing as mp
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import flax.linen as nn
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import polars as pl
import transformers
from orbax import checkpoint as orbax_checkpoint
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. InChI to SMILES conversion will not work.")

# Import MD4 modules
from md4 import rdkit_utils, sampling, train, utils
from md4.configs.md4 import molecular


def process_single_molecule(args_tuple):
    """Worker function for multiprocessing molecule conversion.
    
    Args:
        args_tuple: (inchi, fingerprint, pad_to_length)
    
    Returns:
        Tuple of (smiles, atom_types, inchi, fingerprint) or None if failed
    """
    inchi, fingerprint, pad_to_length = args_tuple
    
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        # Convert InChI to SMILES
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
        
        # Extract features
        features = rdkit_utils.process_smiles(smiles, fp_radius=2, fp_bits=2048, pad_to_length=pad_to_length)
        if features is None or 'atom_types' not in features:
            return None
            
        return (smiles, features['atom_types'], inchi, fingerprint)
    except Exception:
        return None


@dataclasses.dataclass
class EvalResults:
    """Container for evaluation results."""
    original_smiles: List[str]
    original_fingerprints: List[np.ndarray]
    generated_smiles: List[List[str]]
    original_inchi: Optional[List[str]] = None
    
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


class MolecularEvaluator:
    """Main evaluator class for molecular generation."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model, self.train_state = self._load_model_and_state()
        self.rng = utils.get_rng(42)  # Fixed seed for reproducible evaluation
        
    def _load_config(self) -> ml_collections.ConfigDict:
        """Load molecular configuration."""
        config = molecular.get_config()
        config.batch_size = self.args.batch_size
        return config
        
    def _load_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        """Load SMILES tokenizer."""
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.tokenizer or self.args.tokenizer_path)
            print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {self.config.tokenizer or self.args.tokenizer_path}: {e}")
            
    def _load_checkpoint_manager(self) -> orbax_checkpoint.CheckpointManager:
        """Load checkpoint manager."""
        if not self.args.checkpoint_dir:
            raise ValueError("checkpoint_dir must be specified when loading checkpoints")
        checkpointers = dict(train_state=orbax_checkpoint.PyTreeCheckpointer())
        return orbax_checkpoint.CheckpointManager(
            self.args.checkpoint_dir,
            checkpointers=checkpointers,
            options=orbax_checkpoint.CheckpointManagerOptions(create=False),
        )
        
    def _load_model_and_state(self) -> Tuple[nn.Module, train.TrainState]:
        """Load model and checkpoint state."""
        rng = utils.get_rng(self.config.seed)
        data_shape = (self.config.max_length,)
        
        # Create dummy schedule function (not used for inference)
        schedule_fn = lambda step: self.config.learning_rate
        
        # Create model and train state
        model, optimizer, train_state, metrics_class = train.create_train_state(
            self.config,
            rng,
            input_shape=(self.config.batch_size * 10,) + data_shape,
            schedule_fn=schedule_fn,
        )
        
        # Skip checkpoint loading if --no_checkpoint flag is used
        if self.args.no_checkpoint:
            print("WARNING: Using random model weights (--no_checkpoint flag)")
            return model, train_state
        
        # Load checkpoint
        checkpoint_manager = self._load_checkpoint_manager()
        step = self.args.checkpoint_step if self.args.checkpoint_step >= 0 else checkpoint_manager.latest_step()
        
        if step is None:
            raise ValueError(f"No checkpoints found in {self.args.checkpoint_dir}")
        
        print(f"Loading checkpoint from step {step}")
        
        checkpointed_state = {"train_state": train_state}
        checkpointed_state = checkpoint_manager.restore(step, items=checkpointed_state)
        train_state = checkpointed_state["train_state"]
        
        return model, train_state

    def _load_eval_data(self) -> pl.DataFrame:
        """Load MSG evaluation dataset."""
        if not os.path.exists(self.args.eval_data):
            raise FileNotFoundError(f"Evaluation data not found: {self.args.eval_data}")
        
        df = pl.read_parquet(self.args.eval_data)
        print(f"Loaded {len(df)} evaluation samples")
        return df



    def _prepare_molecular_data_iterator(self, eval_df: pl.DataFrame, pad_to_length: int = 128, 
                                       num_processes: Optional[int] = None) -> Iterator[Tuple[str, np.ndarray, str, np.ndarray]]:
        """Convert InChI data to SMILES and extract features using multiprocessing.
        
        Args:
            eval_df: DataFrame containing InChI data
            pad_to_length: Length to pad atom types to
            num_processes: Number of processes to use (None for auto)
            
        Yields:
            Tuples of (smiles, atom_types, inchi, fingerprint) for successfully processed molecules
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required to convert InChI to SMILES.")
        
        columns = eval_df.columns
        inchi_columns = [col for col in columns if 'inchi' in col.lower()]
        
        if not inchi_columns:
            raise ValueError("No InChI columns found in the dataframe")
        
        inchi_col = inchi_columns[0]
        print(f"Found InChI data in column: {inchi_col}")
        
        inchi_list = eval_df[inchi_col].to_list()
        fingerprints_list = eval_df["predicted_fingerprint"].to_list() if "predicted_fingerprint" in eval_df.columns else []
        
        if not fingerprints_list:
            print("WARNING: No predicted_fingerprint column found, using zero fingerprints")
            fingerprints_list = [np.zeros(2048) for _ in range(len(inchi_list))]
        
        print(f"Converting {len(inchi_list)} InChI to SMILES using multiprocessing...")
        
        # Prepare arguments for multiprocessing
        args_list = [(inchi, fingerprints_list[i], pad_to_length) for i, inchi in enumerate(inchi_list)]
        
        # Use multiprocessing to convert molecules
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(args_list))
        
        print(f"Using {num_processes} processes for molecular conversion")
        processed_count = 0
        total_count = len(args_list)
        
        with mp.Pool(processes=num_processes) as pool:
            # Use imap_unordered for better memory efficiency and immediate results
            for result in tqdm(pool.imap_unordered(process_single_molecule, args_list, chunksize=max(1, len(args_list) // (num_processes * 4))), 
                             total=total_count, desc="Processing molecules"):
                if result is not None:
                    smiles, atom_types, inchi, fingerprint = result
                    processed_count += 1
                    yield smiles, atom_types, inchi, fingerprint
        
        print(f"Successfully processed {processed_count}/{total_count} molecules")

    def _prepare_molecular_data(self, eval_df: pl.DataFrame, pad_to_length: int = 128) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
        """Convert InChI data to SMILES and extract features (legacy method for debug mode)."""
        smiles_list = []
        atom_types_list = []
        inchi_list = []
        fingerprints_list = []
        
        for smiles, atom_types, inchi, fingerprint in self._prepare_molecular_data_iterator(eval_df, pad_to_length, num_processes=self.args.num_processes):
            smiles_list.append(smiles)
            atom_types_list.append(atom_types)
            inchi_list.append(inchi)
            fingerprints_list.append(fingerprint)
        
        return smiles_list, atom_types_list, inchi_list, fingerprints_list

    def _process_fingerprints(self, fingerprints: np.ndarray, threshold: float = 0.5, fold_factor: int = 2) -> np.ndarray:
        """Process fingerprints by thresholding and folding."""
        if not RDKIT_AVAILABLE:
            print("WARNING: RDKit not available, skipping fingerprint processing")
            return fingerprints
        
        try:
            # Convert to binary with threshold
            binary_fingerprints = (fingerprints >= threshold).astype(np.int32)
            
            # Fold the fingerprints
            first_half = binary_fingerprints[:, :2048]
            second_half = binary_fingerprints[:, 2048:]
            folded_fingerprints = np.logical_or(first_half, second_half).astype(np.int32)
            return folded_fingerprints
            
        except Exception as e:
            print(f"WARNING: Failed to process fingerprints: {e}")
            return fingerprints

    def _generate_single_datapoint(self, fingerprint: np.ndarray, atom_types: np.ndarray) -> List[str]:
        """Generate samples for a single datapoint using simple_generate."""
        # Process fingerprint
        processed_fp = self._process_fingerprints(fingerprint[None, :], threshold=0.5, fold_factor=2)[0]
        
        # Prepare conditioning
        conditioning = {
            "fingerprint": jnp.repeat(jnp.array(processed_fp, dtype=jnp.int32)[None, :], self.args.num_samples, axis=0),
            "atom_types": jnp.repeat(jnp.array(atom_types, dtype=jnp.int32)[None, :], self.args.num_samples, axis=0),
        }
        
        # Generate samples
        self.rng, sample_rng = jax.random.split(self.rng)
        samples = sampling.simple_generate(
            sample_rng,
            self.train_state,
            self.args.num_samples,
            self.model,
            conditioning=conditioning
        )
        
        # Convert to SMILES strings
        generated_smiles = []
        for i in range(self.args.num_samples):
            tokens = samples[i]
            smiles = self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            generated_smiles.append(smiles)
        
        return generated_smiles

    def _generate_batch_samples(self, fingerprints: np.ndarray, atom_types: np.ndarray) -> List[List[str]]:
        """Generate samples for a batch using pmap."""
        batch_size = fingerprints.shape[0]
        
        # Process fingerprints
        processed_fingerprints = self._process_fingerprints(fingerprints, threshold=0.5, fold_factor=2)
        
        # Expand for multiple samples per input
        expanded_fingerprints = jnp.repeat(jnp.array(processed_fingerprints, dtype=jnp.int32), self.args.num_samples, axis=0)
        expanded_atom_types = jnp.repeat(jnp.array(atom_types, dtype=jnp.int32), self.args.num_samples, axis=0)

        total_samples = batch_size * self.args.num_samples
        dummy_inputs = jnp.ones((total_samples, self.config.max_length), dtype="int32")
        
        conditioning = {
            "fingerprint": expanded_fingerprints,
            "atom_types": expanded_atom_types,
        }
        
        # Handle multi-device distribution
        num_devices = jax.device_count()
        per_device_batch_size = total_samples // num_devices
        
        if total_samples % num_devices != 0:
            padding_needed = num_devices - (total_samples % num_devices)
            dummy_inputs = jnp.concatenate([
                dummy_inputs,
                jnp.ones((padding_needed, self.config.max_length), dtype="int32")
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
            padding_needed = 0
        
        # Reshape for pmap
        dummy_inputs = dummy_inputs.reshape(num_devices, per_device_batch_size, self.config.max_length)
        conditioning = {
            "fingerprint": expanded_fingerprints.reshape(num_devices, per_device_batch_size, -1),
            "atom_types": expanded_atom_types.reshape(num_devices, per_device_batch_size, -1),
        }
        
        # Generate samples
        replicated_train_state = flax_utils.replicate(self.train_state)
        replicated_rng = flax_utils.replicate(self.rng)
        
        samples = sampling.generate(
            self.model,
            replicated_train_state,
            replicated_rng,
            dummy_inputs,
            conditioning=conditioning,
        )
        
        # Process results
        samples = flax_utils.unreplicate(samples)
        samples = samples.reshape(-1, self.config.max_length)
        
        if padding_needed > 0:
            samples = samples[:-padding_needed]
        
        samples = samples.reshape(batch_size, self.args.num_samples, -1)
        
        # Convert to SMILES
        batch_results = []
        for batch_idx in range(batch_size):
            sample_list = []
            for sample_idx in range(self.args.num_samples):
                tokens = samples[batch_idx, sample_idx]
                smiles = self.tokenizer.decode(tokens, skip_special_tokens=True)
                sample_list.append(smiles)
            batch_results.append(sample_list)
        
        return batch_results

    def _process_single_molecule_debug(self, eval_df: pl.DataFrame, pad_to_length: int = 128) -> Optional[Tuple[str, np.ndarray, str, np.ndarray]]:
        """Process a single molecule for debug mode - finds first valid molecule."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required to convert InChI to SMILES.")
        
        columns = eval_df.columns
        inchi_columns = [col for col in columns if 'inchi' in col.lower()]
        
        if not inchi_columns:
            raise ValueError("No InChI columns found in the dataframe")
        
        inchi_col = inchi_columns[0]
        print(f"Found InChI data in column: {inchi_col}")
        
        inchi_list = eval_df[inchi_col].to_list()
        fingerprints_list = eval_df["predicted_fingerprint"].to_list() if "predicted_fingerprint" in eval_df.columns else []
        
        if not fingerprints_list:
            print("WARNING: No predicted_fingerprint column found, using zero fingerprints")
            fingerprints_list = [np.zeros(2048) for _ in range(len(inchi_list))]
        
        print(f"Searching for first valid molecule from {len(inchi_list)} InChI entries...")
        
        # Process molecules one by one until we find a valid one
        for i, inchi in enumerate(inchi_list):
            try:
                # Convert InChI to SMILES
                mol = Chem.MolFromInchi(inchi)
                if mol is None:
                    continue
                smiles = Chem.MolToSmiles(mol)
                
                # Extract features
                features = rdkit_utils.process_smiles(smiles, fp_radius=2, fp_bits=2048, pad_to_length=pad_to_length)
                if features is None or 'atom_types' not in features:
                    continue
                    
                fingerprint = fingerprints_list[i] if i < len(fingerprints_list) else np.zeros(2048)
                print(f"Found valid molecule at index {i}")
                return smiles, features['atom_types'], inchi, fingerprint
                
            except Exception as e:
                print(f"Failed to process molecule {i}: {e}")
                continue
        
        return None

    def run_debug_mode(self):
        """Run evaluation in debug mode (single datapoint)."""
        print("=== DEBUG MODE: Processing single datapoint ===")
        
        if self.args.no_checkpoint:
            print("NOTE: Using random model weights - generated samples may be low quality")
        
        eval_df = self._load_eval_data()
        
        # Process only until we find the first valid molecule
        result = self._process_single_molecule_debug(eval_df)
        
        if result is None:
            print("ERROR: No valid molecules found in dataset")
            return
        
        original_smiles, atom_types, original_inchi, fingerprint = result
        fingerprint = np.array(fingerprint)
        atom_types = np.array(atom_types)
        
        print(f"Original InChI: {original_inchi}")
        print(f"Original SMILES: {original_smiles}")
        print(f"Fingerprint shape: {fingerprint.shape}")
        print(f"Atom types shape: {atom_types.shape}")
        print(f"Generating {self.args.num_samples} samples...")
        
        # Generate samples
        generated_smiles = self._generate_single_datapoint(fingerprint, atom_types)
        
        print("\n=== RESULTS ===")
        for i, smiles in enumerate(generated_smiles):
            print(f"Sample {i+1}: {smiles}")
        
        # Save results if output file specified
        if self.args.output_file:
            results_data = {
                "original_inchi": [original_inchi] * len(generated_smiles),
                "original_smiles": [original_smiles] * len(generated_smiles),
                "generated_smiles": generated_smiles,
                "sample_idx": list(range(len(generated_smiles)))
            }
            
            df = pl.DataFrame(results_data)
            os.makedirs(os.path.dirname(self.args.output_file), exist_ok=True)
            df.write_csv(self.args.output_file)
            print(f"\nResults saved to: {self.args.output_file}")

    def _save_batch_results(self, batch_idx: int, batch_smiles: List[str], batch_fingerprints: np.ndarray, 
                           batch_generated: List[List[str]], batch_inchi: Optional[List[str]] = None):
        """Save results for a single batch."""
        os.makedirs(self.args.intermediate_dir, exist_ok=True)
        
        batch_data = {
            "original_smiles": [],
            "generated_smiles": [],
        }
        
        if batch_inchi is not None:
            batch_data["original_inchi"] = []
        
        for i in range(len(batch_smiles)):
            for j, gen_smi in enumerate(batch_generated[i]):
                batch_data["original_smiles"].append(batch_smiles[i])
                batch_data["generated_smiles"].append(gen_smi)
                if batch_inchi is not None:
                    batch_data["original_inchi"].append(batch_inchi[i])

        batch_df = pl.DataFrame(batch_data)
        batch_file = os.path.join(self.args.intermediate_dir, f"batch_{batch_idx:04d}.csv")
        batch_df.write_csv(batch_file)
        print(f"Saved batch {batch_idx} results to {batch_file}")

    def _combine_intermediate_results(self):
        """Combine all intermediate batch CSV files into final output."""
        if not os.path.exists(self.args.intermediate_dir):
            print(f"WARNING: Intermediate directory {self.args.intermediate_dir} not found")
            return
        
        batch_files = sorted([
            f for f in os.listdir(self.args.intermediate_dir) 
            if f.startswith("batch_") and f.endswith(".csv")
        ])
        
        if not batch_files:
            print(f"WARNING: No batch files found in {self.args.intermediate_dir}")
            return
        
        print(f"Combining {len(batch_files)} batch files into {self.args.output_file}")
        
        all_dfs = []
        for batch_file in batch_files:
            batch_path = os.path.join(self.args.intermediate_dir, batch_file)
            batch_df = pl.read_csv(batch_path)
            all_dfs.append(batch_df)
        
        combined_df = pl.concat(all_dfs)
        os.makedirs(os.path.dirname(self.args.output_file), exist_ok=True)
        combined_df.write_csv(self.args.output_file)
        
        print(f"Combined results saved to {self.args.output_file}")
        print(f"Total samples: {len(combined_df)}")

    def run_batch_mode(self):
        """Run evaluation in batch mode using streaming processing."""
        print("=== BATCH MODE: Processing full dataset with streaming ===")
        
        if self.args.no_checkpoint:
            print("NOTE: Using random model weights - generated samples may be low quality")
        
        eval_df = self._load_eval_data()
        batch_size = self.config.batch_size
        
        print(f"Batch size: {batch_size}")
        print(f"Intermediate results will be saved to: {self.args.intermediate_dir}")
        
        # Collect molecules into batches and process them as they become available
        current_batch = []
        batch_idx = 0
        total_processed = 0
        
        molecule_iterator = self._prepare_molecular_data_iterator(eval_df, num_processes=self.args.num_processes)
        
        try:
            for smiles, atom_types, inchi, fingerprint in molecule_iterator:
                current_batch.append((smiles, atom_types, inchi, fingerprint))
                
                # Process batch when full
                if len(current_batch) == batch_size:
                    self._process_and_save_batch(current_batch, batch_idx)
                    total_processed += len(current_batch)
                    batch_idx += 1
                    current_batch = []
            
            # Process remaining incomplete batch if any
            if current_batch:
                print(f"Processing final incomplete batch of {len(current_batch)} molecules")
                self._process_and_save_batch(current_batch, batch_idx)
                total_processed += len(current_batch)
                batch_idx += 1
            
        except KeyboardInterrupt:
            print(f"Processing interrupted. Processed {total_processed} molecules in {batch_idx} batches.")
            
        print(f"Processed {total_processed} molecules in {batch_idx} batches total")
        
        # Combine all results
        self._combine_intermediate_results()
        print("Batch evaluation completed successfully!")

    def _process_and_save_batch(self, batch_data: List[Tuple[str, np.ndarray, str, np.ndarray]], batch_idx: int):
        """Process a single batch of molecules and save results."""
        if not batch_data:
            return
            
        # Unpack batch data
        batch_smiles, batch_atom_types, batch_inchi, batch_fingerprints = zip(*batch_data)
        batch_fingerprints = np.array(batch_fingerprints)
        batch_atom_types = np.array(list(batch_atom_types))
        
        print(f"Processing batch {batch_idx} with {len(batch_data)} molecules...")
        
        # Generate samples for batch
        batch_generated = self._generate_batch_samples(batch_fingerprints, batch_atom_types)
        
        # Save batch results
        self._save_batch_results(batch_idx, list(batch_smiles), batch_fingerprints, batch_generated, list(batch_inchi))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MSG dataset evaluation script")
    
    # Required arguments
    parser.add_argument("--checkpoint_dir", help="Directory containing checkpoints (required unless --no_checkpoint is used)")
    parser.add_argument("--mode", choices=["debug", "batch"], required=True, 
                       help="Evaluation mode: 'debug' for single datapoint, 'batch' for full dataset")
    
    # Data arguments
    parser.add_argument("--eval_data", default="./data/msg/msg_processed.parquet", 
                       help="Path to MSG eval dataset")
    parser.add_argument("--tokenizer_path", default="data/smiles_tokenizer", 
                       help="Path to SMILES tokenizer")
    
    # Output arguments
    parser.add_argument("--output_file", default="./results/msg_eval_results.csv", 
                       help="Output file for results")
    parser.add_argument("--intermediate_dir", default="./results/intermediate", 
                       help="Directory for intermediate batch results")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="Number of SMILES to generate per data point")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for generation (batch mode only)")
    parser.add_argument("--checkpoint_step", type=int, default=-1, 
                       help="Checkpoint step to load (-1 for latest)")
    parser.add_argument("--no_checkpoint", action="store_true", 
                       help="Skip loading model checkpoint (use random weights)")
    
    # Processing arguments
    parser.add_argument("--num_processes", type=int, default=None, 
                       help="Number of processes for molecular data processing (default: auto)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate inputs
    if not args.no_checkpoint:
        if not args.checkpoint_dir:
            raise ValueError("Must specify --checkpoint_dir or use --no_checkpoint")
        if not os.path.exists(args.checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    elif args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        print(f"WARNING: Checkpoint directory {args.checkpoint_dir} not found, but --no_checkpoint flag is used")
    
    if not os.path.exists(args.eval_data):
        raise ValueError(f"Evaluation data not found: {args.eval_data}")
    
    mode_info = f"{args.mode} mode"
    if args.no_checkpoint:
        mode_info += " (no checkpoint)"
    print(f"Starting MSG evaluation in {mode_info}...")
    
    # Create evaluator and run
    evaluator = MolecularEvaluator(args)
    
    if args.mode == "debug":
        evaluator.run_debug_mode()
    else:  # batch mode
        evaluator.run_batch_mode()


if __name__ == "__main__":
    main()
