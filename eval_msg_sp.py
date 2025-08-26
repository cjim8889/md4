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
import functools
import importlib.util
import sys
from typing import Iterator, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import polars as pl
from etils import epath
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from orbax import checkpoint as orbax_checkpoint
from tqdm import tqdm

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. InChI to SMILES conversion will not work.")

# Import MD4 modules
from md4 import sampling
from md4.input_pipeline_pubchem_large_text import SentencePieceTokenizer
from md4.utils import checkpoint_utils, rdkit_utils, state_utils, utils


def load_config_from_path(config_path: str) -> ml_collections.ConfigDict:
    """Load configuration from a given path.

    Args:
        config_path: Path to the config file. Can be either:
                    - Module path like "md4.configs.md4.molecular_xtra_large"
                    - File path like "/path/to/config.py"
                    - Relative path like "md4/configs/md4/molecular_xtra_large.py"

    Returns:
        Configuration object
    """
    print(f"Loading config from: {config_path}")

    # Handle different path formats
    if config_path.startswith("md4/") and config_path.endswith(".py"):
        # Relative path from md4 package - convert to module path
        # e.g., "md4/configs/md4/molecular_xtra_large.py" -> "md4.configs.md4.molecular_xtra_large"
        module_path = config_path.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]  # Remove .py extension

        # Import as module
        try:
            module = importlib.import_module(module_path)
            return module.get_config()
        except ImportError as e:
            raise ImportError(f"Could not import config module {module_path}") from e
    elif (
        "." in config_path
        and not config_path.startswith("/")
        and not config_path.endswith(".py")
    ):
        # Direct module path like "md4.configs.md4.molecular_xtra_large"
        try:
            module = importlib.import_module(config_path)
            return module.get_config()
        except ImportError as e:
            raise ImportError(f"Could not import config module {config_path}") from e
    else:
        # Absolute file path
        config_file = epath.Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load module from file path
        spec = importlib.util.spec_from_file_location("config_module", str(config_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from {config_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = module
        spec.loader.exec_module(module)
        return module.get_config()


def smiles_to_molecular_formula(smiles_list: List[str]) -> List[str]:
    """Convert SMILES strings to molecular formulas using RDKit.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of molecular formulas (e.g., "C14H19NO2")
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to convert SMILES to molecular formulas.")

    formulas = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
            if mol is not None:
                formula = Chem.rdMolDescriptors.CalcMolFormula(mol)  # type: ignore[attr-defined]
                formulas.append(formula)
            else:
                # Fallback to empty formula if parsing fails
                formulas.append("")
        except Exception:
            # Fallback to empty formula if any error occurs
            formulas.append("")

    return formulas


def safe_decode_tokens(tokenizer, tokens) -> str:
    """Safely decode tokens using SentencePiece tokenizer.

    Args:
        tokenizer: SentencePieceTokenizer instance
        tokens: Token sequence (can be numpy array, list, or tensor)

    Returns:
        Decoded string
    """
    try:
        # Convert to list of integers
        if hasattr(tokens, "tolist"):
            tokens_list = tokens.tolist()
        elif hasattr(tokens, "numpy"):
            tokens_list = tokens.numpy().tolist()
        else:
            tokens_list = list(tokens)

        # Use batch_decode method like in train.py for consistency
        texts = tokenizer.batch_decode(
            [tokens_list],  # Convert single sequence to batch format
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        return texts[0]  # Return the first (and only) decoded text

    except Exception as e:
        print(
            f"Warning: Failed to decode tokens {tokens[:10] if len(tokens) > 10 else tokens}: {e}"
        )
        return ""


def extract_smiles_between_sep(decoded_text: str) -> str:
    """Extract SMILES content after [SEP] token and remove spaces.

    Args:
        decoded_text: Decoded text from tokenizer. SentencePiece automatically removes
                     [BEGIN], [END], and [PAD] tokens, so the format is: "formula [SEP] smiles"

    Returns:
        Clean SMILES string with spaces removed, or empty string if parsing fails

    Example:
        Input: "C21H19N3O2S2 [SEP] CS 1 (=O) (C 2)C1 c1ccc(N n2 nc( cc3 cs c(- c4ccccc4) c23)cc1"
        Output: "CS1(=O)(C2)C1c1ccc(Nn2nc(cc3csc(-c4ccccc4)c23)cc1"
    """
    try:
        # Split by [SEP] and take the last part (more robust approach)
        parts = decoded_text.split("[SEP]")

        # We need at least 2 parts (formula and SMILES)
        if len(parts) < 2:
            return ""

        # Take the last part as SMILES (handles cases with multiple [SEP] tokens)
        smiles_content = parts[-1].strip()

        # Remove all spaces
        smiles_content = smiles_content.replace(" ", "")

        return smiles_content

    except Exception:
        # Return empty string if any parsing error occurs
        return ""


def tokenize_smiles_with_formulas(
    tokenizer, smiles_list: List[str], max_length: int
) -> dict:
    """Tokenize SMILES with corresponding molecular formulas using SentencePiece tokenizer.

    This function aligns with the training pipeline's tokenization approach.

    Args:
        tokenizer: The SentencePieceTokenizer instance
        smiles_list: List of SMILES strings
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs
    """
    # Generate molecular formulas from SMILES
    formulas = smiles_to_molecular_formula(smiles_list)

    # Create combined text exactly as in training pipeline
    # Format: "formula[SEP]smiles" (SentencePiece will automatically add [BEGIN] and [END])
    combined_texts = []
    for formula, smiles in zip(formulas, smiles_list):
        combined_text = f"{formula}[SEP]{smiles}"
        combined_texts.append(combined_text)

    # Tokenize using the same approach as training pipeline's _tokenize_and_truncate
    input_ids = []
    for text in combined_texts:
        # Use tokenizer.encode directly (it handles BOS/EOS automatically)
        tokens = tokenizer.encode(text)

        # Convert to numpy array if it's a tensor
        if hasattr(tokens, "numpy"):
            tokens = tokens.numpy()

        # Ensure it's a list for manipulation
        tokens = list(tokens)

        # Truncate to max_length (same as training)
        tokens = tokens[:max_length]

        # Pad to max_length with pad_id (same as training)
        current_length = len(tokens)
        padding_length = max_length - current_length
        tokens.extend([int(tokenizer.pad_id)] * padding_length)

        input_ids.append(tokens)

    return {"input_ids": np.array(input_ids, dtype=np.int32)}


class MolecularEvaluator:
    """Main evaluator class for molecular generation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Set up output paths
        self._setup_output_paths()

        # Create mesh for evaluation
        self.mesh, mesh_config = self._create_mesh()

        self.config = self._load_config()
        # Override config's mesh configuration with our evaluation mesh
        self.config.mesh_config = mesh_config
        self.tokenizer = self._load_tokenizer()
        self.model, self.train_state, self.state_sharding = self._load_model_and_state()
        self.rng = utils.get_rng(42)  # Fixed seed for reproducible evaluation

        # Create data sharding for generation
        self.data_sharding = NamedSharding(self.mesh, P("data", None))
        
        # Create JIT-compiled generation function (following sharded_train_v2.py pattern)
        # The actual batch size for generation is batch_size * num_samples
        self.jit_generate = jax.jit(
            functools.partial(
                sampling.simple_generate,
                batch_size=self.config.batch_size * self.args.num_samples,
                model=self.model,
                use_conditional_init=self.args.use_conditional_init,
            ),
            in_shardings=(None, self.state_sharding, self.data_sharding, self.data_sharding),
            out_shardings=(self.data_sharding),
        )

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
        """Set up output file paths from output_dir."""
        # Create main output directory if it doesn't exist
        output_dir = epath.Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up output file path
        self.output_file = output_dir / "msg_eval_results.csv"
        # Keep intermediate dir for batch processing
        self.intermediate_dir = output_dir / "intermediate"

    def _create_mesh(self):
        """Create device mesh for evaluation."""
        # Create mesh configuration for evaluation (single data axis with 8 devices)
        mesh_config = ml_collections.ConfigDict(
            {"mesh_shape": (8,), "mesh_axis_names": ("data",)}
        )

        # Create the mesh
        mesh = jax.make_mesh(mesh_config.mesh_shape, mesh_config.mesh_axis_names)
        print(f"Created evaluation mesh: {mesh}")

        return mesh, mesh_config

    def _load_config(self) -> ml_collections.ConfigDict:
        """Load molecular configuration."""
        # Use config path from args if provided, otherwise default to molecular_xtra_large
        config_path = getattr(
            self.args, "config_path", "md4.configs.md4.molecular_xtra_large"
        )
        config = load_config_from_path(config_path)
        config.batch_size = self.args.batch_size
        return config

    def _load_tokenizer(self):
        """Load SentencePiece SMILES tokenizer."""
        try:
            tokenizer_path = epath.Path(self.config.tokenizer or self.args.tokenizer_path)
            # Handle .model file extension for SentencePiece
            if not str(tokenizer_path).endswith(".model"):
                tokenizer_path = tokenizer_path / "sentencepiece_tokenizer.model"
                if not tokenizer_path.exists():
                    # Try with different common paths
                    possible_paths = [
                        epath.Path(str(self.config.tokenizer or self.args.tokenizer_path) + ".model"),
                        epath.Path("data/sentencepiece_tokenizer.model"),
                        epath.Path("data/sentencepiece_tokenizer_4096_bpe_latest.model"),
                    ]
                    for path in possible_paths:
                        if path.exists():
                            tokenizer_path = path
                            break

            tokenizer = SentencePieceTokenizer(
                model_path=str(tokenizer_path), add_bos=True, add_eos=True
            )
            print(
                f"Loaded SentencePiece tokenizer with vocab size: {tokenizer.vocab_size} from {tokenizer_path}"
            )
            return tokenizer
        except Exception as e:
            raise ValueError(
                f"Failed to load SentencePiece tokenizer from {self.config.tokenizer or self.args.tokenizer_path}: {e}"
            )

    def _load_checkpoint_manager(self) -> orbax_checkpoint.CheckpointManager:
        """Load checkpoint manager."""
        if not self.args.checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must be specified when loading checkpoints"
            )

        return checkpoint_utils.get_checkpoint_manager(
            self.config, self.args.checkpoint_dir, create=False
        )

    def _load_model_and_state(self) -> Tuple[nn.Module, state_utils.TrainState, jax.sharding.NamedSharding]:
        """Load model and checkpoint state with sharding."""
        rng = utils.get_rng(int(self.config.seed))  # type: ignore
        data_shape = (int(self.config.max_length),)  # type: ignore

        # Create dummy schedule function (not used for inference)
        def schedule_fn(step):
            return self.config.learning_rate

        # Ensure mesh is available
        if self.mesh is None:
            raise RuntimeError("Mesh must be initialized before loading model and state")

        # Use the mesh created during initialization
        mesh = self.mesh

        # Initialize sharding
        data_sharding = NamedSharding(mesh, P("data", None))

        # Create sharded model and train state (matching sharded_train_v2.py)
        with mesh:
            model, _, train_state, _, state_sharding = (
                state_utils.create_sharded_train_state(
                    self.config,
                    data_sharding,
                    mesh,
                    rng,  # type: ignore
                    input_shape=(int(self.config.batch_size),) + data_shape,  # type: ignore
                    schedule_fn=schedule_fn,
                )
            )

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

        # For sharded checkpoints, restore within mesh context
        with mesh:
            train_state = checkpoint_manager.restore(
                step, args=orbax_checkpoint.args.StandardRestore(train_state)
            )
            # Place train_state with appropriate sharding
            train_state = jax.device_put(train_state, state_sharding)

        return model, train_state, state_sharding

    def _load_eval_data(self) -> pl.DataFrame:
        """Load MSG evaluation dataset."""
        eval_data_path = epath.Path(self.args.eval_data)
        if not eval_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.args.eval_data}")

        df = pl.read_parquet(str(eval_data_path))
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
            fingerprints_list = [
                np.zeros(self.args.fp_bits) for _ in range(len(inchi_list))
            ]

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
                smiles = Chem.MolToSmiles(mol)  # type: ignore[attr-defined]

                # Extract features
                features = rdkit_utils.process_smiles(
                    smiles, fp_radius=2, fp_bits=self.args.fp_bits
                )
                if features is None:
                    continue

                # Get fingerprints
                predicted_fingerprint = (
                    np.array(fingerprints_list[i], dtype=np.float32)
                    if i < len(fingerprints_list)
                    else np.zeros(self.args.fp_bits, dtype=np.float32)
                )
                original_fingerprint = (
                    features
                    if isinstance(features, np.ndarray)
                    else np.zeros(self.args.fp_bits)
                )

                processed_count += 1
                # Use original dataset index as unique ID - this ensures each datapoint has a unique identifier
                # even if InChI values are the same but predicted_fingerprints are different
                yield i, smiles, inchi, predicted_fingerprint, original_fingerprint

            except Exception:
                # Skip molecules that fail to process
                continue

        print(f"Successfully processed {processed_count}/{total_count} molecules")


    def _generate_samples(
        self,
        predicted_fingerprints: np.ndarray,
        original_fingerprints: np.ndarray,
        smiles_list: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Unified generation method that handles both single and multi-device scenarios."""
        # Ensure fingerprints are 2D (batch_size, fingerprint_dim)
        if predicted_fingerprints.ndim == 1:
            predicted_fingerprints = predicted_fingerprints[None, :]
        if original_fingerprints.ndim == 1:
            original_fingerprints = original_fingerprints[None, :]

        batch_size = predicted_fingerprints.shape[0]

        # Choose conditioning fingerprints based on fp_mode
        if self.args.fp_mode == "original":
            fingerprints_for_conditioning = original_fingerprints
        else:
            # Use predicted fingerprints (default)
            fingerprints_for_conditioning = predicted_fingerprints

        # Repeat per requested samples
        expanded_fingerprints = jnp.repeat(
            jnp.array(fingerprints_for_conditioning, dtype=jnp.float32),
            self.args.num_samples,
            axis=0,
        )

        total_samples = batch_size * self.args.num_samples

        # Optional conditional init from SMILES
        tokens_repeated = None
        if (
            self.args.use_conditional_init
            and smiles_list is not None
        ):
            # Use paired tokenization with molecular formulas and SMILES
            enc = tokenize_smiles_with_formulas(
                self.tokenizer,
                smiles_list,
                int(self.config.max_length),  # type: ignore
            )
            tokens = jnp.asarray(enc["input_ids"], dtype=jnp.int32)  # (B, L)
            tokens_repeated = jnp.repeat(tokens, self.args.num_samples, axis=0)

        # Prepare conditioning
        conditioning = {"cross_conditioning": expanded_fingerprints}

        # Use sharded generation (required)
        if not (self.jit_generate and self.state_sharding and self.data_sharding):
            raise RuntimeError("Sharded generation is required but not properly initialized")
        
        # Use sharded generation within mesh context
        print(f"Using sharded generation for {total_samples} samples")
        
        self.rng, sample_rng = jax.random.split(self.rng)
        
        # Generate using the JIT-compiled function
        with self.mesh:
            samples = self.jit_generate(
                sample_rng,
                self.train_state,
                conditioning,
                tokens_repeated,
            )
        
        if isinstance(samples, tuple):
            samples = samples[0]
        
        # Ensure samples is properly shaped
        if samples.ndim == 2:  # (total_samples, max_length)
            # Reshape to (batch_size, num_samples_per_input, max_length)
            actual_samples_per_input = samples.shape[0] // batch_size
            samples = samples.reshape(batch_size, actual_samples_per_input, -1)
        elif samples.ndim == 3:  # Already (batch_size, num_samples, max_length)
            actual_samples_per_input = samples.shape[1]
        else:
            raise ValueError(f"Unexpected samples shape: {samples.shape}")

        # Convert to SMILES strings
        batch_results = []
        for b in range(batch_size):
            if batch_size == 1 and actual_samples_per_input <= 10:
                # Debug-friendly printout
                sample_list = []
                for i in range(actual_samples_per_input):
                    tokens = samples[b, i]
                    print(f"Generated tokens: {tokens}")
                    decoded_text = safe_decode_tokens(self.tokenizer, tokens)
                    print(f"Raw decoded: {decoded_text}")
                    clean_smiles = extract_smiles_between_sep(decoded_text)
                    print(f"Clean SMILES: {clean_smiles}")
                    sample_list.append(clean_smiles)
            else:
                # Use safe decoding for SentencePiece tokenizer
                sample_list = []
                for i in range(actual_samples_per_input):
                    tokens = samples[b, i]
                    decoded_text = safe_decode_tokens(self.tokenizer, tokens)
                    clean_smiles = extract_smiles_between_sep(decoded_text)
                    sample_list.append(clean_smiles)
            batch_results.append(sample_list)

        return batch_results


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
        intermediate_dir = epath.Path(self.intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

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
        batch_file = intermediate_dir / f"batch_{batch_idx:04d}.csv"
        batch_df.write_csv(str(batch_file))
        # print(f"Saved batch {batch_idx} results to {batch_file}")


    def _process_and_save_batch(
        self,
        batch_data: List[Tuple[int, str, str, np.ndarray, np.ndarray]],
        batch_idx: int,
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
            batch_predicted_fingerprints,
            batch_original_fingerprints,
            list(batch_smiles) if self.args.use_conditional_init else None,
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

    def run(self):
        """Run evaluation using streaming processing."""
        print("=== Processing full dataset with streaming ===")

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
                    (
                        eval_id,
                        smiles,
                        inchi,
                        predicted_fingerprint,
                        original_fingerprint,
                    )
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

        # Combine all intermediate results into final output
        intermediate_dir = epath.Path(self.intermediate_dir)
        if intermediate_dir.exists():
            batch_files = sorted(
                [
                    f.name
                    for f in intermediate_dir.iterdir()
                    if f.name.startswith("batch_") and f.name.endswith(".csv")
                ]
            )
            
            if batch_files:
                print(f"Combining {len(batch_files)} batch files into {self.output_file}")
                all_dfs = []
                for batch_file in batch_files:
                    batch_path = intermediate_dir / batch_file
                    batch_df = pl.read_csv(str(batch_path))
                    all_dfs.append(batch_df)
                
                combined_df = pl.concat(all_dfs)
                output_file = epath.Path(self.output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                combined_df.write_csv(str(output_file))
                
                print(f"Combined results saved to {self.output_file}")
                print(f"Total samples: {len(combined_df)}")
        
        print("Evaluation completed successfully!")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MSG dataset evaluation script")

    # Required arguments
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing checkpoints",
    )

    # Data arguments
    parser.add_argument(
        "--eval_data",
        default="./data/msg/msg_processed.parquet",
        help="Path to MSG eval dataset",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="data/sentencepiece_tokenizer.model",
        help="Path to SentencePiece SMILES tokenizer model file",
    )
    parser.add_argument(
        "--config_path",
        default="md4.configs.md4.molecular_xtra_large",
        help="Path to model configuration. Can be module path (md4.configs.md4.molecular_xtra_large) or file path",
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
        help="Batch size for generation",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help="Checkpoint step to load (-1 for latest)",
    )

    # Processing arguments
    parser.add_argument(
        "--fp_mode",
        choices=["original", "predicted"],
        default="predicted",
        help="Which fingerprints to use for generation: 'predicted' (default) or 'original'",
    )
    parser.add_argument(
        "--fp_bits",
        type=int,
        default=2048,
        help="Number of bits for fingerprint generation (default: 2048)",
    )
    parser.add_argument(
        "--use_conditional_init",
        action="store_true",
        help="Initialize zt from input SMILES using model.conditional_sample (masks after first token id 3)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Validate inputs
    checkpoint_dir = epath.Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    eval_data = epath.Path(args.eval_data)
    if not eval_data.exists():
        raise ValueError(f"Evaluation data not found: {args.eval_data}")

    # Build info string
    mode_info = f"MSG evaluation (using {args.fp_mode} fingerprints)"
    print(f"Starting {mode_info}...")

    # Create evaluator and run
    evaluator = MolecularEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
