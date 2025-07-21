#!/usr/bin/env python3
"""Analysis script for MSG evaluation results - refactored for efficiency and clarity.

Usage:
    # Full analysis
    python analyze_msg_results.py --results_file=./results/msg_eval_results.csv
    
    # Debug mode (only 10 samples)
    python analyze_msg_results.py --results_file=./results/msg_eval_results.csv --debug
    
    # From intermediate batch files
    python analyze_msg_results.py --intermediate_dir=./results/intermediate/
"""

import os
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from absl import app, flags, logging

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, Descriptors, inchi, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Analysis will be limited.")

FLAGS = flags.FLAGS
flags.DEFINE_string("results_file", "./results/msg_eval_results.csv", "Path to evaluation results (CSV or parquet)")
flags.DEFINE_string("intermediate_dir", "", "Directory containing intermediate batch CSV files (alternative to results_file)")
flags.DEFINE_string("output_dir", "./results/analysis/", "Directory to save analysis results")
flags.DEFINE_string("overlapping_prefixes_file", "", "Path to CSV file containing overlapping InChI key prefixes to filter out")
flags.DEFINE_bool("plot_figures", True, "Whether to generate plots")
flags.DEFINE_bool("molecular_grid", True, "Whether to generate molecular structure grid")
flags.DEFINE_bool("debug", False, "Debug mode: process only 10 datapoints for quick testing")


@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    validity: Dict
    uniqueness: Dict  
    similarity: Dict
    properties: Dict
    

def load_overlapping_prefixes(file_path: str) -> Set[str]:
    """Load overlapping InChI key prefixes to filter out."""
    if not file_path or not os.path.exists(file_path):
        return set()
    
    try:
        df = pl.read_csv(file_path)
        prefixes = set(df.select("overlapping_prefix").to_series().to_list())
        logging.info(f"Loaded {len(prefixes)} overlapping prefixes")
        return prefixes
    except Exception as e:
        logging.warning(f"Failed to load overlapping prefixes: {e}")
        return set()


def load_and_combine_data(results_file: str = "", intermediate_dir: str = "") -> pl.DataFrame:
    """Load data from either results file or intermediate directory."""
    if intermediate_dir:
        batch_files = sorted([f for f in os.listdir(intermediate_dir) 
                            if f.startswith("batch_") and f.endswith(".csv")])
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {intermediate_dir}")
        
        logging.info(f"Loading {len(batch_files)} batch files")
        dfs = [pl.read_csv(os.path.join(intermediate_dir, f)) for f in batch_files]
        return pl.concat(dfs)
    
    if results_file.endswith('.parquet'):
        return pl.read_parquet(results_file)
    return pl.read_csv(results_file)


def create_mol_objects(smiles_list: list) -> Dict[str, Optional[object]]:
    """Create RDKit mol objects once and cache them."""
    if not RDKIT_AVAILABLE:
        return {}
    
    mol_cache = {}
    for smiles in smiles_list:
        try:
            mol_cache[smiles] = Chem.MolFromSmiles(smiles) if smiles else None
        except:
            mol_cache[smiles] = None
    return mol_cache


def prepare_molecular_data(df: pl.DataFrame, overlapping_prefixes: Set[str]) -> pl.DataFrame:
    """Single function to prepare all molecular data with caching."""
    logging.info("Preparing molecular data...")
    
    # Clean spaces from generated SMILES
    df = df.with_columns(
        pl.col("generated_smiles").str.replace_all(" ", "").alias("generated_smiles")
    )
    
    # Get all unique SMILES for mol object creation
    all_smiles = set()
    all_smiles.update(df["original_smiles"].unique().to_list())
    all_smiles.update(df["generated_smiles"].unique().to_list())
    all_smiles.discard(None)
    all_smiles.discard("")
    
    # Create mol objects once
    mol_cache = create_mol_objects(list(all_smiles))
    
    def get_canonical_smiles(smiles: str) -> Optional[str]:
        mol = mol_cache.get(smiles)
        try:
            return Chem.CanonSmiles(smiles) if mol else None
        except:
            return None
    
    def is_valid_mol(smiles: str) -> bool:
        return mol_cache.get(smiles) is not None
    
    def get_inchi_key_prefix(smiles: str) -> str:
        mol = mol_cache.get(smiles)
        try:
            return inchi.MolToInchiKey(mol)[:14] if mol else ""
        except:
            return ""
    
    def calculate_properties(smiles: str) -> Dict[str, float]:
        mol = mol_cache.get(smiles)
        if not mol:
            return {}
        try:
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
            }
        except:
            return {}
    
    def calculate_tanimoto(smiles1: str, smiles2: str) -> float:
        mol1, mol2 = mol_cache.get(smiles1), mol_cache.get(smiles2)
        if not (mol1 and mol2):
            return 0.0
        try:
            fpgen = AllChem.GetRDKitFPGenerator()
            fp1, fp2 = fpgen.GetFingerprint(mol1), fpgen.GetFingerprint(mol2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0
    
    # Add all computed columns at once
    df = df.with_columns([
        # Canonical SMILES
        pl.col("original_smiles").map_elements(get_canonical_smiles, return_dtype=pl.String).alias("original_canonical"),
        pl.col("generated_smiles").map_elements(get_canonical_smiles, return_dtype=pl.String).alias("generated_canonical"),
        
        # Validity
        pl.col("original_smiles").map_elements(is_valid_mol, return_dtype=pl.Boolean).alias("original_valid"),
        pl.col("generated_smiles").map_elements(is_valid_mol, return_dtype=pl.Boolean).alias("generated_valid"),
        
        # InChI key prefix for filtering
        pl.col("original_smiles").map_elements(get_inchi_key_prefix, return_dtype=pl.String).alias("inchi_prefix")
    ])
    
    # Filter overlapping samples
    if overlapping_prefixes:
        original_count = len(df)
        df = df.filter(~pl.col("inchi_prefix").is_in(list(overlapping_prefixes)))
        logging.info(f"Filtered out {original_count - len(df)} overlapping samples")
    
    # Add sample IDs if missing
    if "sample_id" not in df.columns:
        unique_originals = df.select("original_smiles").unique().with_row_index("sample_id")
        df = df.join(unique_originals, on="original_smiles", how="left")
    
    # Calculate similarity for valid pairs
    def calc_similarity_safe(row):
        return calculate_tanimoto(row["original_smiles"], row["generated_smiles"]) if row["original_valid"] and row["generated_valid"] else None
    
    df = df.with_columns(
        pl.struct(["original_smiles", "generated_smiles", "original_valid", "generated_valid"])
        .map_elements(calc_similarity_safe, return_dtype=pl.Float64)
        .alias("tanimoto_similarity")
    )
    
    # Calculate properties for unique valid molecules
    unique_valid_original = df.filter(pl.col("original_valid")).select("original_smiles").unique()
    unique_valid_generated = df.filter(pl.col("generated_valid")).select("generated_smiles").unique()
    
    # Store properties as JSON strings to avoid complex nested structures in polars
    original_props = {row["original_smiles"]: calculate_properties(row["original_smiles"]) 
                     for row in unique_valid_original.iter_rows(named=True)}
    generated_props = {row["generated_smiles"]: calculate_properties(row["generated_smiles"]) 
                      for row in unique_valid_generated.iter_rows(named=True)}
    
    # Store properties in a way that can be used later
    df = df.with_columns([
        pl.col("original_smiles").map_elements(lambda s: original_props.get(s, {}), return_dtype=pl.Object).alias("original_props"),
        pl.col("generated_smiles").map_elements(lambda s: generated_props.get(s, {}), return_dtype=pl.Object).alias("generated_props")
    ])
    
    logging.info(f"Prepared {len(df)} samples with molecular data")
    return df


def analyze_validity(df: pl.DataFrame) -> Dict:
    """Analyze validity and exact matches."""
    total_count = len(df)
    valid_count = df.filter(pl.col("generated_valid")).height
    
    # Count unique original molecules that have at least one exact match
    molecules_with_exact_matches = (df.filter(
        pl.col("original_valid") & 
        pl.col("generated_valid") &
        (pl.col("original_canonical") == pl.col("generated_canonical"))
    )
    .select("sample_id")  # Use sample_id to ensure we count unique original molecules
    .unique()
    .height)
    
    # Total unique original molecules for denominator
    total_unique_molecules = df.select("sample_id").unique().height
    
    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "validity_rate": valid_count / total_count if total_count > 0 else 0,
        "exact_match_count": molecules_with_exact_matches,
        "exact_match_rate": molecules_with_exact_matches / total_unique_molecules if total_unique_molecules > 0 else 0,
        "total_unique_molecules": total_unique_molecules,
    }


def analyze_uniqueness(df: pl.DataFrame) -> Dict:
    """Analyze global and per-sample uniqueness."""
    valid_df = df.filter(pl.col("generated_valid"))
    total_valid = len(valid_df)
    
    # Global uniqueness
    unique_global = valid_df.select("generated_canonical").unique().height
    global_uniqueness = unique_global / total_valid if total_valid > 0 else 0
    
    # Per-sample uniqueness
    per_sample_stats = (valid_df
                       .group_by("sample_id")
                       .agg([
                           pl.count().alias("total_in_sample"),
                           pl.col("generated_canonical").n_unique().alias("unique_in_sample")
                       ])
                       .with_columns(
                           (pl.col("unique_in_sample") / pl.col("total_in_sample")).alias("uniqueness_ratio")
                       ))
    
    avg_per_sample = per_sample_stats["uniqueness_ratio"].mean()
    
    return {
        "total_valid": total_valid,
        "unique_generated": unique_global,
        "global_uniqueness": global_uniqueness,
        "avg_per_sample_uniqueness": avg_per_sample,
    }


def analyze_similarity(df: pl.DataFrame) -> Dict:
    """Analyze Tanimoto similarity distribution."""
    similarities = df.filter(pl.col("tanimoto_similarity").is_not_null())["tanimoto_similarity"].to_list()
    
    if not similarities:
        return {"avg_similarity": 0.0, "similarities": []}
    
    return {
        "avg_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "max_similarity": np.max(similarities),
        "min_similarity": np.min(similarities),
        "similarities": similarities,
    }


def analyze_properties(df: pl.DataFrame) -> Dict:
    """Analyze molecular property distributions."""
    if not RDKIT_AVAILABLE:
        return {}
    
    # Get unique valid molecules and their properties
    unique_original = (df.filter(pl.col("original_valid"))
                      .group_by("original_smiles")
                      .first()
                      .select(["original_props"]))
    
    valid_generated = df.filter(pl.col("generated_valid")).select(["generated_props"])
    
    # Extract property values
    prop_names = ["molecular_weight", "logp", "tpsa", "num_atoms", "num_bonds", "num_rings"]
    results = {}
    
    for prop in prop_names:
        orig_values = [props.get(prop) for props in unique_original["original_props"].to_list() 
                      if props and prop in props]
        gen_values = [props.get(prop) for props in valid_generated["generated_props"].to_list() 
                     if props and prop in props]
        
        if orig_values and gen_values:
            results[prop] = {
                "original_mean": np.mean(orig_values),
                "original_std": np.std(orig_values), 
                "generated_mean": np.mean(gen_values),
                "generated_std": np.std(gen_values),
                "original_values": orig_values,
                "generated_values": gen_values,
            }
    
    return results


def create_summary_plot(results: AnalysisResults, output_dir: str):
    """Create summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MSG Evaluation Summary", fontsize=16)
    
    validity = results.validity
    uniqueness = results.uniqueness
    similarity = results.similarity
    properties = results.properties
    
    # Validity (samples) and Exact Matches (molecules)
    validity_categories = ["Valid Samples", "Invalid Samples"]
    validity_counts = [validity["valid_count"], validity["total_count"] - validity["valid_count"]]
    
    # Create a dual bar chart to show both sample validity and molecule exact matches
    x_pos = np.arange(len(validity_categories))
    axes[0,0].bar(x_pos - 0.2, validity_counts, 0.4, color=["green", "red"], alpha=0.7, label="Samples")
    
    # Add exact match bar (scaled to show proportion)
    exact_match_categories = ["Exact Match Molecules", "Other Molecules"] 
    exact_match_counts = [validity["exact_match_count"], 
                         validity["total_unique_molecules"] - validity["exact_match_count"]]
    axes[0,0].bar([2.2, 2.6], exact_match_counts, 0.4, color=["gold", "lightgray"], alpha=0.7, label="Molecules")
    
    axes[0,0].set_xticks([0, 1, 2.4])
    axes[0,0].set_xticklabels(["Valid", "Invalid", "Exact Match"])
    axes[0,0].set_title(f"Validity: {validity['validity_rate']:.1%} | Exact: {validity['exact_match_rate']:.1%}")
    axes[0,0].legend(fontsize=8)
    
    # Uniqueness  
    axes[0,1].bar(["Unique", "Duplicate"],
                 [uniqueness["unique_generated"], 
                  uniqueness["total_valid"] - uniqueness["unique_generated"]],
                 color=["blue", "orange"])
    axes[0,1].set_title(f"Uniqueness: {uniqueness['global_uniqueness']:.1%}")
    
    # Similarity
    if similarity["similarities"]:
        axes[1,0].hist(similarity["similarities"], bins=20, alpha=0.7, color="purple")
        axes[1,0].axvline(similarity["avg_similarity"], color="red", linestyle="--")
        axes[1,0].set_title(f"Tanimoto Similarity (μ={similarity['avg_similarity']:.3f})")
    
    # Properties
    if "molecular_weight" in properties:
        mw = properties["molecular_weight"]
        axes[1,1].boxplot([mw["original_values"], mw["generated_values"]], 
                         tick_labels=["Original", "Generated"])
        axes[1,1].set_title("Molecular Weight")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_molecular_grid(df: pl.DataFrame, output_dir: str, max_samples: int = 10):
    """Create molecular comparison grid."""
    if not RDKIT_AVAILABLE:
        return
        
    # Get samples with high similarity for visualization
    sample_data = (df.filter(pl.col("original_valid") & pl.col("generated_valid") & pl.col("tanimoto_similarity").is_not_null())
                  .group_by("sample_id")
                  .agg([
                      pl.first("original_smiles").alias("original"),
                      pl.col("generated_smiles").head(5).alias("generated"),
                      pl.col("tanimoto_similarity").head(5).alias("similarities")
                  ])
                  .head(max_samples))
    
    fig, axes = plt.subplots(max_samples, 6, figsize=(18, 3*max_samples))
    if max_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, row in enumerate(sample_data.iter_rows(named=True)):
        # Original molecule
        orig_mol = Chem.MolFromSmiles(row["original"])
        if orig_mol:
            img = Draw.MolToImage(orig_mol, size=(200, 200))
            axes[i,0].imshow(img)
            axes[i,0].set_title("Original", fontweight="bold")
            axes[i,0].axis('off')
        
        # Generated molecules
        for j, (gen_smiles, sim) in enumerate(zip(row["generated"], row["similarities"])):
            if j < 5:
                gen_mol = Chem.MolFromSmiles(gen_smiles)
                if gen_mol:
                    img = Draw.MolToImage(gen_mol, size=(200, 200))
                    axes[i,j+1].imshow(img)
                    color = "green" if sim >= 0.7 else "orange" if sim >= 0.5 else "red"
                    axes[i,j+1].set_title(f"Sim: {sim:.3f}", color=color)
                    axes[i,j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "molecular_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_report(results: AnalysisResults, output_dir: str):
    """Generate text summary report."""
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write("MSG Evaluation Report\n")
        f.write("="*20 + "\n")
        if FLAGS.debug:
            f.write("*** DEBUG MODE - LIMITED DATASET (10 molecules) ***\n")
        f.write("\n")
        
        # Validity
        v = results.validity
        f.write(f"Sample validity: {v['valid_count']}/{v['total_count']} ({v['validity_rate']:.1%})\n")
        f.write(f"Exact matches: {v['exact_match_count']}/{v['total_unique_molecules']} molecules ({v['exact_match_rate']:.1%})\n")
        f.write(f"  Note: Exact matches counted once per unique original molecule\n\n")
        
        # Uniqueness
        u = results.uniqueness
        f.write(f"Global uniqueness: {u['unique_generated']}/{u['total_valid']} ({u['global_uniqueness']:.1%})\n")
        f.write(f"Avg per-sample uniqueness: {u['avg_per_sample_uniqueness']:.1%}\n\n")
        
        # Similarity
        s = results.similarity
        if s["similarities"]:
            f.write(f"Tanimoto similarity: {s['avg_similarity']:.3f} ± {s['std_similarity']:.3f}\n\n")
        
        # Properties
        for prop, stats in results.properties.items():
            f.write(f"{prop}: orig={stats['original_mean']:.2f}±{stats['original_std']:.2f}, "
                   f"gen={stats['generated_mean']:.2f}±{stats['generated_std']:.2f}\n")


def main(argv):
    del argv
    
    mode_info = "DEBUG MODE (10 samples)" if FLAGS.debug else "FULL ANALYSIS"
    logging.info(f"Starting MSG analysis - {mode_info}...")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Load and prepare data
    overlapping_prefixes = load_overlapping_prefixes(FLAGS.overlapping_prefixes_file)
    df = load_and_combine_data(FLAGS.results_file, FLAGS.intermediate_dir)
    
    # Debug mode: limit to first 10 samples
    if FLAGS.debug:
        original_count = len(df)
        # Get first 10 unique original molecules and all their generated samples
        unique_originals = df.select("original_smiles").unique().head(10)
        df = df.join(unique_originals, on="original_smiles", how="inner")
        logging.info(f"Debug mode: Limited dataset from {original_count} to {len(df)} samples "
                    f"({len(unique_originals)} unique molecules)")
    
    df = prepare_molecular_data(df, overlapping_prefixes)
    
    # Run all analyses
    results = AnalysisResults(
        validity=analyze_validity(df),
        uniqueness=analyze_uniqueness(df), 
        similarity=analyze_similarity(df),
        properties=analyze_properties(df)
    )
    
    # Log key results
    if FLAGS.debug:
        logging.info("=== DEBUG MODE RESULTS ===")
        logging.info(f"Processed {len(df)} samples from {df.select('sample_id').unique().height} molecules")
    
    logging.info(f"Validity: {results.validity['validity_rate']:.1%} ({results.validity['valid_count']}/{results.validity['total_count']} samples)")
    logging.info(f"Exact matches: {results.validity['exact_match_rate']:.1%} ({results.validity['exact_match_count']}/{results.validity['total_unique_molecules']} molecules)")
    logging.info(f"Uniqueness: {results.uniqueness['global_uniqueness']:.1%} ({results.uniqueness['unique_generated']}/{results.uniqueness['total_valid']})")
    logging.info(f"Avg similarity: {results.similarity['avg_similarity']:.3f}")
    if results.properties:
        logging.info(f"Properties calculated for {len(results.properties)} metrics")
    
    # Generate outputs
    if FLAGS.plot_figures:
        create_summary_plot(results, FLAGS.output_dir)
    
    if FLAGS.molecular_grid:
        create_molecular_grid(df, FLAGS.output_dir)
    
    generate_report(results, FLAGS.output_dir)
    
    mode_suffix = " (DEBUG MODE - limited dataset)" if FLAGS.debug else ""
    logging.info(f"Analysis complete{mode_suffix}. Results saved to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main) 