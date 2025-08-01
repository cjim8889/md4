#!/usr/bin/env python3
"""Evaluation script for MSG dataset: load checkpoints and generate SMILES.

The script assigns unique eval_id to each datapoint in the evaluation dataset.
This ensures that even if multiple entries have the same InChI but different 
predicted fingerprints, they are treated as separate evaluation instances.

Output CSV format:
- eval_id: Unique identifier for each evaluation datapoint (from original dataset index)
- original_inchi: Original InChI string
- original_smiles: SMILES converted from InChI
- generated_smiles: Generated SMILES from the model
- sample_idx: Index of the generated sample (0 to num_samples-1)

Each eval_id will have num_samples rows in the output, one for each generated molecule.
"""

import argparse
import os
from typing import Iterator, List, Optional, Tuple

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
from md4 import rdkit_utils, sampling, utils, state_utils
from md4.configs.md4 import molecular_finetune


class MolecularEvaluator:
    """Main evaluator class for molecular generation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Set up output paths
        self._setup_output_paths()

        # For combine mode, we only need basic setup
        if args.mode == "combine":
            return

        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model, self.train_state = self._load_model_and_state()
        self.rng = utils.get_rng(42)  # Fixed seed for reproducible evaluation
        
        # Pre-replicate training state for multi-device generation to avoid replicating it every batch
        # Since training state doesn't change during inference, we can replicate it once at startup
        # and reuse it for all batches, providing significant performance improvement
        self.replicated_train_state = None
        if jax.device_count() > 1:
            self.replicated_train_state = flax_utils.replicate(self.train_state)

        # Debug: Check if parameters are loaded correctly
        print("Checking parameter values after checkpoint loading...")
        print(f"Train state step: {self.train_state.step}")

        # Print parameter structure to understand the layout
        print("Parameter structure:")
        print(jax.tree_util.tree_map(lambda x: x.shape, self.train_state.params))

        # Check a few parameter values to see if they're reasonable (not all zeros)
        try:
            # Find any parameter to check - let's be more flexible about the structure
            def check_params(params, name):
                print(f"\nChecking {name} parameters:")
                # Get a flat list of all parameter arrays
                leaves = jax.tree_util.tree_leaves(params)
                if leaves:
                    first_param = leaves[0]
                    print(f"  First param shape: {first_param.shape}")
                    print(f"  First param mean: {jnp.mean(first_param):.6f}")
                    print(f"  First param std: {jnp.std(first_param):.6f}")
                    print(
                        f"  First param min/max: {jnp.min(first_param):.6f}/{jnp.max(first_param):.6f}"
                    )

                    # Check if all values are zero or very close to zero
                    is_zero = jnp.allclose(first_param, 0.0, atol=1e-8)
                    print(f"  Is essentially zero: {is_zero}")
                else:
                    print(f"  No parameters found in {name}")

            check_params(self.train_state.params, "regular")

            if self.train_state.ema_params is not None:
                check_params(self.train_state.ema_params, "EMA")
            else:
                print("No EMA params found!")

        except Exception as e:
            print(f"Error checking parameters: {e}")
            print(
                "Parameter keys:",
                list(self.train_state.params.keys())
                if hasattr(self.train_state.params, "keys")
                else "No keys method",
            )

        # Log EMA availability
        if self.train_state.ema_params is not None:
            print("EMA weights available - will be used for generation")
        else:
            print("WARNING: No EMA params available, using regular params")

    def _setup_output_paths(self):
        """Set up output file and intermediate directory paths from output_dir."""
        # Create main output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Set up derived paths
        self.output_file = os.path.join(self.args.output_dir, "msg_eval_results.csv")
        self.intermediate_dir = os.path.join(self.args.output_dir, "intermediate")

    def _load_config(self) -> ml_collections.ConfigDict:
        """Load molecular configuration."""
        config = molecular_finetune.get_config()
        config.batch_size = self.args.batch_size
        return config

    def _load_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        """Load SMILES tokenizer."""
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.tokenizer or self.args.tokenizer_path
            )
            print(
                f"Loaded tokenizer with vocab size: {tokenizer.vocab_size} from {self.config.tokenizer or self.args.tokenizer_path}"
            )
            return tokenizer
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from {self.config.tokenizer or self.args.tokenizer_path}: {e}"
            )

    def _load_checkpoint_manager(self) -> orbax_checkpoint.CheckpointManager:
        """Load checkpoint manager."""
        if not self.args.checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must be specified when loading checkpoints"
            )
        checkpointers = dict(train_state=orbax_checkpoint.PyTreeCheckpointer())
        return orbax_checkpoint.CheckpointManager(
            self.args.checkpoint_dir,
            checkpointers=checkpointers,
            options=orbax_checkpoint.CheckpointManagerOptions(create=False),
        )

    def _load_model_and_state(self) -> Tuple[nn.Module, state_utils.TrainState]:
        """Load model and checkpoint state."""
        rng = utils.get_rng(self.config.seed)
        data_shape = (self.config.max_length,)

        # Create dummy schedule function (not used for inference)
        def schedule_fn(step):
            return self.config.learning_rate

        # Create model and train state
        model, _, train_state, _ = state_utils.create_train_state(
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
        step = (
            self.args.checkpoint_step
            if self.args.checkpoint_step >= 0
            else checkpoint_manager.latest_step()
        )

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

    def _prepare_molecular_data_iterator(
        self, eval_df: pl.DataFrame
    ) -> Iterator[Tuple[int, str, str, np.ndarray, np.ndarray]]:
        """Convert InChI data to SMILES and extract features sequentially.

        Args:
            eval_df: DataFrame containing InChI data

        Yields:
            Tuples of (unique_id, smiles, inchi, predicted_fingerprint, original_fingerprint) for successfully processed molecules
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required to convert InChI to SMILES.")

        columns = eval_df.columns
        inchi_columns = [col for col in columns if "inchi" in col.lower()]

        if not inchi_columns:
            raise ValueError("No InChI columns found in the dataframe")

        inchi_col = inchi_columns[0]
        print(f"Found InChI data in column: {inchi_col}")

        inchi_list = eval_df[inchi_col].to_list()
        fingerprints_list = (
            eval_df["predicted_fingerprint"].to_list()
            if "predicted_fingerprint" in eval_df.columns
            else []
        )

        if not fingerprints_list:
            print(
                "WARNING: No predicted_fingerprint column found, using zero fingerprints"
            )
            fingerprints_list = [np.zeros(2048) for _ in range(len(inchi_list))]

        print(f"Converting {len(inchi_list)} InChI to SMILES...")

        processed_count = 0
        total_count = len(inchi_list)

        # Process molecules sequentially with progress bar
        for i, inchi in enumerate(tqdm(inchi_list, desc="Processing molecules")):
            try:
                # Convert InChI to SMILES
                mol = Chem.MolFromInchi(inchi)
                if mol is None:
                    continue
                smiles = Chem.MolToSmiles(mol)

                # Extract features
                features = rdkit_utils.process_smiles(smiles, fp_radius=2, fp_bits=2048)
                if features is None:
                    continue

                # Get fingerprints
                predicted_fingerprint = (
                    np.array(fingerprints_list[i], dtype=np.float32)
                    if i < len(fingerprints_list)
                    else np.zeros(2048, dtype=np.float32)
                )
                original_fingerprint = (
                    features if isinstance(features, np.ndarray) else np.zeros(2048)
                )

                processed_count += 1
                # Use original dataset index as unique ID - this ensures each datapoint has a unique identifier
                # even if InChI values are the same but predicted_fingerprints are different
                yield i, smiles, inchi, predicted_fingerprint, original_fingerprint

            except Exception:
                # Skip molecules that fail to process
                continue

        print(f"Successfully processed {processed_count}/{total_count} molecules")

    def _process_fingerprints(
        self,
        predicted_fingerprints: np.ndarray,
        original_fingerprints: np.ndarray,
        threshold: float = 0.5,
        mode: str = "or",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process fingerprints by thresholding and folding, and calculate bit differences.

        Args:
            predicted_fingerprints: Predicted fingerprints from model
            original_fingerprints: Original fingerprints computed from SMILES
            threshold: Threshold for binarization
            mode: Folding mode - 'or', 'xor', or 'and'

        Returns:
            Tuple of (processed_fingerprints, bit_differences)
        """
        if not RDKIT_AVAILABLE:
            print("WARNING: RDKit not available, skipping fingerprint processing")
            return predicted_fingerprints, np.zeros(
                predicted_fingerprints.shape[0]
                if predicted_fingerprints.ndim > 1
                else 1
            )

        if mode == "none":
            return predicted_fingerprints, np.zeros(
                predicted_fingerprints.shape[0]
                if predicted_fingerprints.ndim > 1
                else 1
            )
        # Validate mode
        if mode not in ["or", "xor", "and"]:
            raise ValueError(
                f"Invalid folding mode '{mode}'. Must be one of: 'or', 'xor', 'and'"
            )

        # Convert predicted fingerprints to binary with threshold
        binary_predicted = (predicted_fingerprints >= threshold).astype(np.int32)

        # Ensure both fingerprints have same shape
        if binary_predicted.ndim == 1:
            binary_predicted = binary_predicted[None, :]
        if original_fingerprints.ndim == 1:
            original_fingerprints = original_fingerprints[None, :]

        # Fold the predicted fingerprints (original should already be folded/processed)
        if binary_predicted.shape[1] > 2048:
            first_half = binary_predicted[:, :2048]
            second_half = binary_predicted[:, 2048:]

            # Apply the specified folding mode
            if mode == "or":
                folded_predicted = np.logical_or(first_half, second_half).astype(
                    np.int32
                )
            elif mode == "xor":
                folded_predicted = np.logical_xor(first_half, second_half).astype(
                    np.int32
                )
            elif mode == "and":
                folded_predicted = np.logical_and(first_half, second_half).astype(
                    np.int32
                )
        else:
            folded_predicted = binary_predicted

        # Calculate bit differences
        bit_differences = np.sum(folded_predicted != original_fingerprints, axis=1)

        return folded_predicted, bit_differences

    def _generate_samples(
        self, predicted_fingerprints: np.ndarray, original_fingerprints: np.ndarray
    ) -> List[List[str]]:
        """Unified generation method that handles both single and multi-device scenarios."""
        # Ensure fingerprints are 2D (batch_size, fingerprint_dim)
        if predicted_fingerprints.ndim == 1:
            predicted_fingerprints = predicted_fingerprints[None, :]
        if original_fingerprints.ndim == 1:
            original_fingerprints = original_fingerprints[None, :]
            
        batch_size = predicted_fingerprints.shape[0]

        # Process fingerprints and get bit differences
        processed_fingerprints, bit_differences = self._process_fingerprints(
            predicted_fingerprints,
            original_fingerprints,
            threshold=0.5,
            mode=self.args.fingerprint_mode,
        )

        # Use the selected fingerprints for generation
        if self.args.use_original_fingerprints:
            fingerprints_for_conditioning = original_fingerprints
        else:
            fingerprints_for_conditioning = processed_fingerprints

        # Expand for multiple samples per input
        expanded_fingerprints = jnp.repeat(
            jnp.array(fingerprints_for_conditioning, dtype=jnp.float32),
            self.args.num_samples,
            axis=0,
        )

        total_samples = batch_size * self.args.num_samples
        dummy_inputs = jnp.ones((total_samples, self.config.max_length), dtype="int32")

        conditioning = {
            "fingerprint": expanded_fingerprints,
        }

        # Handle device distribution
        num_devices = jax.device_count()
        
        # Check if we should use pmap (multi-device) or simple generation (single device)
        use_pmap = num_devices > 1 and total_samples >= num_devices and self.replicated_train_state is not None
        
        if use_pmap:
            # Multi-device path using pmap
            # print(f"Using pmap with {num_devices} devices for generation")
            
            # Adjust total_samples to be divisible by num_devices to avoid padding
            samples_per_device = total_samples // num_devices
            if total_samples % num_devices != 0:
                # Drop the remainder to avoid padding issues
                adjusted_total_samples = samples_per_device * num_devices
                print(f"Adjusting batch size from {total_samples} to {adjusted_total_samples} to fit {num_devices} devices")
                
                # Trim inputs and conditioning to the adjusted size
                dummy_inputs = dummy_inputs[:adjusted_total_samples]
                expanded_fingerprints = expanded_fingerprints[:adjusted_total_samples]
                total_samples = adjusted_total_samples
            
            per_device_batch_size = total_samples // num_devices

            # Reshape for pmap
            dummy_inputs = dummy_inputs.reshape(
                num_devices, per_device_batch_size, self.config.max_length
            )
            conditioning = {
                "fingerprint": expanded_fingerprints.reshape(
                    num_devices, per_device_batch_size, -1
                ),
            }

            # Generate samples using pmap - use pre-replicated training state
            self.rng, sample_rng = jax.random.split(self.rng)
            replicated_rng = flax_utils.replicate(sample_rng)

            samples = sampling.generate(
                self.model,
                self.replicated_train_state,
                replicated_rng,
                dummy_inputs,
                conditioning=conditioning,
            )

            # Collect results from all devices using all_gather
            all_samples = jax.pmap(
                lambda x: jax.lax.all_gather(x, "batch"), axis_name="batch"
            )(samples)
            samples = flax_utils.unreplicate(all_samples)
            samples = samples.reshape(-1, self.config.max_length)
            
        else:
            # Single device path using simple_generate or vmap
            print(f"Using single device generation for {total_samples} samples")
            
            # For EMA params, we need to create appropriate variables
            if self.train_state.ema_params is not None:
                variables = {
                    'params': self.train_state.ema_params,
                    **self.train_state.state
                }
            else:
                variables = {
                    'params': self.train_state.params,
                    **self.train_state.state
                }
            
            self.rng, sample_rng = jax.random.split(self.rng)
            
            # Generate initial noise
            zt = self.model.apply(
                variables,
                total_samples,
                method=self.model.prior_sample,
                rngs={'sample': sample_rng},
            )
            
            self.rng, sample_rng = jax.random.split(self.rng)

            # Diffusion sampling loop
            def body_fn(i, zt):
                return self.model.apply(
                    variables,
                    sample_rng,
                    i,
                    self.model.timesteps,
                    zt,
                    conditioning=conditioning,
                    method=self.model.sample_step,
                )

            z0 = jax.lax.fori_loop(
                lower=0, upper=self.model.timesteps, body_fun=body_fn, init_val=zt
            )
            
            # Decode to tokens
            samples = self.model.apply(
                variables,
                z0,
                conditioning=conditioning,
                method=self.model.decode,
                rngs={'sample': self.rng},
            )
            
            # samples should be tokens directly from decode method
            # If it's a tuple, extract the first element
            if isinstance(samples, tuple):
                samples = samples[0]

        # Reshape to (batch_size, num_samples_per_input, max_length)
        actual_samples_per_input = total_samples // batch_size
        samples = samples.reshape(batch_size, actual_samples_per_input, -1)

        # Convert to SMILES
        batch_results = []
        for batch_idx in range(batch_size):
            sample_list = []
            if batch_size == 1 and actual_samples_per_input <= 10:
                # For debug mode, decode individually and show tokens
                for i in range(actual_samples_per_input):
                    tokens = samples[batch_idx, i]
                    print(f"Generated tokens: {tokens}")
                    smiles = self.tokenizer.decode(
                        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True
                    )
                    sample_list.append(smiles)
            else:
                # For batch mode, decode in batch
                smiles = self.tokenizer.batch_decode(
                    samples[batch_idx], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                sample_list.extend(smiles)
            batch_results.append(sample_list)

        return batch_results

    def _process_single_molecule_debug(
        self, eval_df: pl.DataFrame
    ) -> Optional[Tuple[int, str, str, np.ndarray, np.ndarray]]:
        """Process a single molecule for debug mode - finds first valid molecule."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required to convert InChI to SMILES.")

        columns = eval_df.columns
        inchi_columns = [col for col in columns if "inchi" in col.lower()]

        if not inchi_columns:
            raise ValueError("No InChI columns found in the dataframe")

        inchi_col = inchi_columns[0]
        print(f"Found InChI data in column: {inchi_col}")

        inchi_list = eval_df[inchi_col].to_list()
        fingerprints_list = (
            eval_df["predicted_fingerprint"].to_list()
            if "predicted_fingerprint" in eval_df.columns
            else []
        )

        if not fingerprints_list:
            print(
                "WARNING: No predicted_fingerprint column found, using zero fingerprints"
            )
            fingerprints_list = [np.zeros(2048) for _ in range(len(inchi_list))]

        print(
            f"Searching for first valid molecule from {len(inchi_list)} InChI entries..."
        )

        # Process molecules one by one until we find a valid one
        for i, inchi in enumerate(inchi_list):
            try:
                # Convert InChI to SMILES
                mol = Chem.MolFromInchi(inchi)
                if mol is None:
                    continue
                smiles = Chem.MolToSmiles(mol)

                # Extract features
                features = rdkit_utils.process_smiles(smiles, fp_radius=2, fp_bits=2048)
                if features is None:
                    continue

                predicted_fingerprint = np.array(fingerprints_list[i], dtype=np.float32)
                original_fingerprint = features
                print(f"Found valid molecule at index {i}")
                return i, smiles, inchi, predicted_fingerprint, original_fingerprint

            except Exception as e:
                print(f"Failed to process molecule {i}: {e}")
                continue

        return None

    def run_debug_mode(self):
        """Run evaluation in debug mode (single datapoint)."""
        print("=== DEBUG MODE: Processing single datapoint ===")

        if self.args.no_checkpoint:
            print(
                "NOTE: Using random model weights - generated samples may be low quality"
            )

        eval_df = self._load_eval_data()

        # Process only until we find the first valid molecule
        result = self._process_single_molecule_debug(eval_df)

        if result is None:
            print("ERROR: No valid molecules found in dataset")
            return

        unique_id, original_smiles, original_inchi, predicted_fingerprint, original_fingerprint = (
            result
        )
        predicted_fingerprint = np.array(predicted_fingerprint)
        original_fingerprint = np.array(original_fingerprint)

        print("=" * 20)
        print(f"Unique ID: {unique_id}")
        print(f"Original InChI: {original_inchi}")
        print(f"Original SMILES: {original_smiles}")
        print(f"Original Fingerprint: {original_fingerprint}")
        print(f"Predicted fingerprint: {predicted_fingerprint}")
        print(f"Predicted fingerprint shape: {predicted_fingerprint.shape}")
        print(f"Original fingerprint shape: {original_fingerprint.shape}")

        print(f"Generating {self.args.num_samples} samples...")

        # Generate samples using unified method
        generated_results = self._generate_samples(
            predicted_fingerprint, original_fingerprint
        )
        generated_smiles = generated_results[0]  # First (and only) item in batch

        print("\n=== RESULTS ===")
        for i, smiles in enumerate(generated_smiles):
            print(f"Sample {i + 1}: {smiles}")

        # Save results if output file specified
        if self.output_file:
            results_data = {
                "eval_id": [unique_id] * len(generated_smiles),
                "original_inchi": [original_inchi] * len(generated_smiles),
                "original_smiles": [original_smiles] * len(generated_smiles),
                "generated_smiles": generated_smiles,
                "sample_idx": list(range(len(generated_smiles))),
            }

            df = pl.DataFrame(results_data)
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            df.write_csv(self.output_file)
            print(f"\nResults saved to: {self.output_file}")

    def _save_batch_results(
        self,
        batch_idx: int,
        batch_eval_ids: List[int],
        batch_smiles: List[str],
        batch_fingerprints: np.ndarray,
        batch_generated: List[List[str]],
        batch_inchi: Optional[List[str]] = None,
    ):
        """Save results for a single batch."""
        os.makedirs(self.intermediate_dir, exist_ok=True)

        batch_data = {
            "eval_id": [],
            "original_smiles": [],
            "generated_smiles": [],
        }

        if batch_inchi is not None:
            batch_data["original_inchi"] = []

        for i in range(len(batch_smiles)):
            for j, gen_smi in enumerate(batch_generated[i]):
                batch_data["eval_id"].append(batch_eval_ids[i])
                batch_data["original_smiles"].append(batch_smiles[i])
                batch_data["generated_smiles"].append(gen_smi)
                if batch_inchi is not None:
                    batch_data["original_inchi"].append(batch_inchi[i])

        batch_df = pl.DataFrame(batch_data)
        batch_file = os.path.join(
            self.intermediate_dir, f"batch_{batch_idx:04d}.csv"
        )
        batch_df.write_csv(batch_file)
        # print(f"Saved batch {batch_idx} results to {batch_file}")

    def _combine_intermediate_results(self):
        """Combine all intermediate batch CSV files into final output."""
        if not os.path.exists(self.intermediate_dir):
            print(
                f"WARNING: Intermediate directory {self.intermediate_dir} not found"
            )
            return

        batch_files = sorted(
            [
                f
                for f in os.listdir(self.intermediate_dir)
                if f.startswith("batch_") and f.endswith(".csv")
            ]
        )

        if not batch_files:
            print(f"WARNING: No batch files found in {self.intermediate_dir}")
            return

        print(f"Combining {len(batch_files)} batch files into {self.output_file}")

        all_dfs = []
        for batch_file in batch_files:
            batch_path = os.path.join(self.intermediate_dir, batch_file)
            batch_df = pl.read_csv(batch_path)
            all_dfs.append(batch_df)

        combined_df = pl.concat(all_dfs)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        combined_df.write_csv(self.output_file)

        print(f"Combined results saved to {self.output_file}")
        print(f"Total samples: {len(combined_df)}")

    def _process_and_save_batch(
        self, batch_data: List[Tuple[int, str, str, np.ndarray, np.ndarray]], batch_idx: int
    ):
        """Process a single batch of molecules and save results."""
        if not batch_data:
            return

        # Unpack batch data
        (
            batch_eval_ids,
            batch_smiles,
            batch_inchi,
            batch_predicted_fingerprints,
            batch_original_fingerprints,
        ) = zip(*batch_data)
        batch_predicted_fingerprints = np.array(batch_predicted_fingerprints)
        batch_original_fingerprints = np.array(batch_original_fingerprints)

        # print(f"Processing batch {batch_idx} with {len(batch_data)} molecules...")

        # Generate samples for batch
        batch_generated = self._generate_samples(
            batch_predicted_fingerprints, batch_original_fingerprints
        )

        # Save batch results
        self._save_batch_results(
            batch_idx,
            list(batch_eval_ids),
            list(batch_smiles),
            batch_predicted_fingerprints,
            batch_generated,
            list(batch_inchi),
        )

    def run_batch_mode(self):
        """Run evaluation in batch mode using streaming processing."""
        print("=== BATCH MODE: Processing full dataset with streaming ===")

        if self.args.no_checkpoint:
            print(
                "NOTE: Using random model weights - generated samples may be low quality"
            )

        eval_df = self._load_eval_data()
        batch_size = self.config.batch_size

        print(f"Batch size: {batch_size}")
        print(f"Intermediate results will be saved to: {self.intermediate_dir}")

        # Collect molecules into batches and process them as they become available
        current_batch = []
        batch_idx = 0
        total_processed = 0

        molecule_iterator = self._prepare_molecular_data_iterator(eval_df)

        try:
            for (
                eval_id,
                smiles,
                inchi,
                predicted_fingerprint,
                original_fingerprint,
            ) in molecule_iterator:
                current_batch.append(
                    (eval_id, smiles, inchi, predicted_fingerprint, original_fingerprint)
                )

                # Process batch when full
                if len(current_batch) == batch_size:
                    self._process_and_save_batch(current_batch, batch_idx)
                    total_processed += len(current_batch)
                    batch_idx += 1
                    current_batch = []

            # Process remaining incomplete batch if any
            if current_batch:
                print(
                    f"Dropping final incomplete batch of {len(current_batch)} molecules"
                )

        except KeyboardInterrupt:
            print(
                f"Processing interrupted. Processed {total_processed} molecules in {batch_idx} batches."
            )

        print(f"Processed {total_processed} molecules in {batch_idx} batches total")

        # Combine all results
        self._combine_intermediate_results()
        print("Batch evaluation completed successfully!")

    def run_combine_mode(self):
        """Run evaluation in combine mode - only combine existing intermediate results."""
        print("=== COMBINE MODE: Combining intermediate results ===")

        if not os.path.exists(self.intermediate_dir):
            raise ValueError(
                f"Intermediate directory not found: {self.intermediate_dir}"
            )

        batch_files = [
            f
            for f in os.listdir(self.intermediate_dir)
            if f.startswith("batch_") and f.endswith(".csv")
        ]

        if not batch_files:
            raise ValueError(f"No batch files found in {self.intermediate_dir}")

        print(f"Found {len(batch_files)} batch files in {self.intermediate_dir}")

        # Combine all results
        self._combine_intermediate_results()
        print("Combine evaluation completed successfully!")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MSG dataset evaluation script")

    # Required arguments
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory containing checkpoints (required unless --no_checkpoint is used)",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "batch", "combine"],
        required=True,
        help="Evaluation mode: 'debug' for single datapoint, 'batch' for full dataset, 'combine' to combine intermediate results",
    )

    # Data arguments
    parser.add_argument(
        "--eval_data",
        default="./data/msg/msg_processed.parquet",
        help="Path to MSG eval dataset",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="data/smiles_tokenizer",
        help="Path to SMILES tokenizer",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Output directory for results (intermediate files will be stored in output_dir/intermediate)",
    )

    # Generation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of SMILES to generate per data point",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (batch mode only)",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help="Checkpoint step to load (-1 for latest)",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Skip loading model checkpoint (use random weights)",
    )

    # Processing arguments
    parser.add_argument(
        "--use_original_fingerprints",
        action="store_true",
        help="Use original RDKit fingerprints instead of predicted fingerprints for generation",
    )
    parser.add_argument(
        "--fingerprint_mode",
        choices=["or", "xor", "and", "none"],
        default="or",
        help="Mode for folding fingerprints when they are longer than 2048 bits: 'or' (default), 'xor', or 'and'",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Validate inputs based on mode
    if args.mode != "combine":
        # For debug and batch modes, validate standard inputs
        if not args.no_checkpoint:
            if not args.checkpoint_dir:
                raise ValueError("Must specify --checkpoint_dir or use --no_checkpoint")
            if not os.path.exists(args.checkpoint_dir):
                raise ValueError(
                    f"Checkpoint directory not found: {args.checkpoint_dir}"
                )
        elif args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
            print(
                f"WARNING: Checkpoint directory {args.checkpoint_dir} not found, but --no_checkpoint flag is used"
            )

        if not os.path.exists(args.eval_data):
            raise ValueError(f"Evaluation data not found: {args.eval_data}")
    else:
        # For combine mode, only validate intermediate directory and output file
        evaluator = MolecularEvaluator(args)  # Need to create evaluator to get paths
        if not os.path.exists(evaluator.intermediate_dir):
            raise ValueError(
                f"Intermediate directory not found: {evaluator.intermediate_dir}"
            )

        batch_files = [
            f
            for f in os.listdir(evaluator.intermediate_dir)
            if f.startswith("batch_") and f.endswith(".csv")
        ]

        if not batch_files:
            raise ValueError(f"No batch files found in {evaluator.intermediate_dir}")
        
        # Don't create evaluator again below
        if args.mode == "debug":
            evaluator.run_debug_mode()
        elif args.mode == "batch":
            evaluator.run_batch_mode()
        elif args.mode == "combine":
            evaluator.run_combine_mode()
        return

    mode_info = f"{args.mode} mode"
    if args.mode != "combine":
        if args.no_checkpoint:
            mode_info += " (no checkpoint)"
        if args.use_original_fingerprints:
            mode_info += " (using original fingerprints)"
        else:
            mode_info += " (using predicted fingerprints)"
        mode_info += f" (fingerprint folding: {args.fingerprint_mode})"
    print(f"Starting MSG evaluation in {mode_info}...")

    # Create evaluator and run (only if not already created for combine mode)
    if args.mode != "combine":
        evaluator = MolecularEvaluator(args)
        
        if args.mode == "debug":
            evaluator.run_debug_mode()
        elif args.mode == "batch":
            evaluator.run_batch_mode()


if __name__ == "__main__":
    main()
