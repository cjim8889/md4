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
from typing import Dict, Set, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from absl import app, flags, logging

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, inchi, Draw, Lipinski, rdMolDescriptors, QED
    # Try to import SAScore - it's not always available
    try:
        from rdkit.Contrib.SA_Score import sascorer
        SASCORE_AVAILABLE = True
    except ImportError:
        SASCORE_AVAILABLE = False
        logging.warning("SAScore not available. Will skip SyntheticAccessibility calculation.")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    SASCORE_AVAILABLE = False
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
        except Exception:
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
        except Exception:
            return None
    
    def is_valid_mol(smiles: str) -> bool:
        return mol_cache.get(smiles) is not None
    
    def get_inchi_key_prefix(smiles: str) -> str:
        mol = mol_cache.get(smiles)
        try:
            return inchi.MolToInchiKey(mol)[:14] if mol else ""
        except Exception:
            return ""
    
    def calculate_properties(smiles: str) -> Dict[str, float]:
        mol = mol_cache.get(smiles)
        if not mol:
            return {}
        try:
            props = {
                'AtomicLogP': Crippen.MolLogP(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'PolarSurfaceArea': rdMolDescriptors.CalcTPSA(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': Lipinski.NumAromaticRings(mol),
                'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
                'FractionCSP3': Lipinski.FractionCSP3(mol),
                'QED': QED.qed(mol)
            }
            
            # BertzCT can cause segmentation faults, so skip it for now
            # try:
            #     props['BertzComplexity'] = rdkit.Chem.GraphDescriptors.BertzCT(mol)
            # except Exception as e:
            #     # Skip BertzComplexity if it fails - some molecules cause crashes
            #     logging.warning(f"BertzCT calculation failed for SMILES (skipping): {e}")
            #     pass
            
            if SASCORE_AVAILABLE:
                try:
                    props['SyntheticAccessibility'] = sascorer.calculateScore(mol)
                except Exception as e:
                    logging.warning(f"SA Score calculation failed for SMILES (skipping): {e}")
                    pass
            return props
        except Exception as e:
            logging.warning(f"Property calculation failed for SMILES (returning empty): {e}")
            return {}
    
    def calculate_tanimoto(smiles1: str, smiles2: str) -> float:
        mol1, mol2 = mol_cache.get(smiles1), mol_cache.get(smiles2)
        if not (mol1 and mol2):
            return 0.0
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
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

    if "eval_id" in df.columns:
        # Use eval_id if available (new format with unique IDs per datapoint)
        df = df.with_columns(pl.col("eval_id").alias("sample_id"))
        logging.info("Using eval_id as sample_id for analysis")
    elif "sample_id" not in df.columns:
        # Fallback: create sample_id from unique original molecules (old format)
        unique_originals = df.select("original_smiles").unique().with_row_index("sample_id")
        df = df.join(unique_originals, on="original_smiles", how="left")
        logging.info("Created sample_id from unique original_smiles (legacy mode)")
    
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
    exact_match_filter = df.filter(
        pl.col("original_valid") & 
        pl.col("generated_valid") &
        (pl.col("original_canonical") == pl.col("generated_canonical"))
    )
    
    # Debug logging for exact matches
    if FLAGS.debug:
        logging.info(f"=== EXACT MATCH DEBUG ===")
        logging.info(f"Total samples: {total_count}")
        logging.info(f"Valid generated samples: {valid_count}")
        logging.info(f"Samples with exact matches: {len(exact_match_filter)}")
        
        # Show some example exact matches
        if len(exact_match_filter) > 0:
            sample_exact_matches = exact_match_filter.head(5)
            logging.info("Example exact matches:")
            for i, row in enumerate(sample_exact_matches.iter_rows(named=True)):
                logging.info(f"  {i+1}. sample_id={row['sample_id']}, original='{row['original_canonical'][:50]}...', generated='{row['generated_canonical'][:50]}...'")
    
    molecules_with_exact_matches = (exact_match_filter
                                   .select("sample_id")  # Use sample_id to ensure we count unique original molecules
                                   .unique()
                                   .height)
    
    # Total unique original molecules for denominator
    total_unique_molecules = df.select("sample_id").unique().height
    
    # Debug logging for the final calculation
    if FLAGS.debug:
        logging.info(f"Unique molecules with exact matches: {molecules_with_exact_matches}")
        logging.info(f"Total unique molecules: {total_unique_molecules}")
        exact_match_rate = molecules_with_exact_matches / total_unique_molecules if total_unique_molecules > 0 else 0
        logging.info(f"Exact match rate: {exact_match_rate:.3f} ({exact_match_rate:.1%})")
        logging.info("=========================\n")
    
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
    """Analyze Tanimoto similarity - per-molecule averages and global statistics."""
    if not RDKIT_AVAILABLE:
        return {"avg_similarity": 0.0, "similarities": []}
    
    # Group by original molecule and calculate average similarity for each
    per_molecule_similarities = (df
        .filter(pl.col("tanimoto_similarity").is_not_null())
        .group_by("sample_id")
        .agg([
            pl.col("tanimoto_similarity").mean().alias("avg_similarity_per_molecule"),
            pl.col("tanimoto_similarity").count().alias("num_valid_generated")
        ])
        .filter(pl.col("num_valid_generated") > 0)
    )
    
    if len(per_molecule_similarities) == 0:
        return {"avg_similarity": 0.0, "similarities": []}
    
    # Get per-molecule averages
    per_molecule_avgs = per_molecule_similarities["avg_similarity_per_molecule"].to_list()
    
    # Also get all individual similarities for distribution analysis
    all_similarities = df.filter(pl.col("tanimoto_similarity").is_not_null())["tanimoto_similarity"].to_list()
    
    return {
        "avg_similarity_per_molecule": np.mean(per_molecule_avgs),  # Average of per-molecule averages
        "std_similarity_per_molecule": np.std(per_molecule_avgs),
        "global_avg_similarity": np.mean(all_similarities),  # Global average across all pairs
        "similarities": all_similarities,
        "per_molecule_similarities": per_molecule_avgs,
        "num_molecules_with_similarities": len(per_molecule_similarities)
    }


def analyze_properties(df: pl.DataFrame) -> Dict:
    """Analyze molecular property distributions with correct per-eval_id MAE calculation."""
    if not RDKIT_AVAILABLE:
        return {}
    
    # Define all property names we want to analyze
    prop_names = ['AtomicLogP', 'NumHAcceptors', 'NumHDonors', 'PolarSurfaceArea', 
                  'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings', 
                  'FractionCSP3', 'QED', 'BertzComplexity']
    
    if SASCORE_AVAILABLE:
        prop_names.append('SyntheticAccessibility')
    
    results = {}
    
    # For each property, calculate MAE per eval_id, then average across eval_ids
    for prop in prop_names:
        eval_id_maes = []  # MAE for each eval_id
        all_original_values = []  # For reporting statistics
        all_generated_values = []  # For reporting statistics
        
        # Debug logging for all properties when in debug mode
        if FLAGS.debug:
            logging.info(f"=== {prop} DEBUG ===")
        
        # Group by eval_id (sample_id)
        grouped = df.group_by("sample_id")
        
        eval_ids_with_no_generated = []
        eval_ids_with_no_original = []
        eval_ids_processed = []
        
        for group_data in grouped:
            sample_id, group_df = group_data
            
            # Get the original molecule properties (should be the same for all rows in this group)
            original_props = None
            generated_prop_values = []
            
            for row in group_df.iter_rows(named=True):
                # Get original properties (same for all rows in this eval_id)
                if original_props is None:
                    orig_props = row.get("original_props", {})
                    if orig_props and prop in orig_props and orig_props[prop] is not None:
                        original_props = orig_props[prop]
                
                # Collect generated properties for this eval_id
                gen_props = row.get("generated_props", {})
                if (gen_props and prop in gen_props and gen_props[prop] is not None and
                    np.isfinite(gen_props[prop])):
                    generated_prop_values.append(gen_props[prop])
            
            # Debug logging for all properties when in debug mode
            if FLAGS.debug:
                if original_props is None:
                    eval_ids_with_no_original.append(sample_id)
                if len(generated_prop_values) == 0:
                    eval_ids_with_no_generated.append(sample_id)
                    
                if len(eval_ids_processed) < 3:  # Show first 3 eval_ids for each property
                    logging.info(f"  eval_id {sample_id}: original={original_props}, generated_values={generated_prop_values}")
                    if len(generated_prop_values) > 0 and original_props is not None:
                        logging.info(f"    -> mean_generated={np.mean(generated_prop_values):.3f}, mae={abs(original_props - np.mean(generated_prop_values)):.3f}")
            
            # Calculate MAE for this eval_id if we have valid data
            if (original_props is not None and len(generated_prop_values) > 0 and
                np.isfinite(original_props)):
                
                # Average of generated properties for this eval_id
                avg_generated = np.mean(generated_prop_values)
                
                # MAE for this eval_id
                eval_id_mae = abs(original_props - avg_generated)
                eval_id_maes.append(eval_id_mae)
                eval_ids_processed.append(sample_id)
                
                # Store for global statistics
                all_original_values.append(original_props)
                all_generated_values.extend(generated_prop_values)
        
        # Debug summary for all properties when in debug mode
        if FLAGS.debug:
            total_eval_ids = len([group for group in grouped])  # Count groups
            logging.info(f"  Total eval_ids processed: {total_eval_ids}")
            logging.info(f"  eval_ids with no original props: {len(eval_ids_with_no_original)} {eval_ids_with_no_original[:3]}")
            logging.info(f"  eval_ids with no generated props: {len(eval_ids_with_no_generated)} {eval_ids_with_no_generated[:3]}")
            logging.info(f"  eval_ids successfully processed: {len(eval_id_maes)}")
            if eval_id_maes:
                logging.info(f"  MAE values for first 5 eval_ids: {[round(mae, 3) for mae in eval_id_maes[:5]]}")
                logging.info(f"  Final mean MAE: {np.mean(eval_id_maes):.3f}")
            logging.info("=========================\n")
        
        if eval_id_maes:  # Only calculate if we have valid eval_ids
            results[prop] = {
                "original_mean": np.mean(all_original_values),
                "original_std": np.std(all_original_values),
                "generated_mean": np.mean(all_generated_values),
                "generated_std": np.std(all_generated_values),
                "mae": np.mean(eval_id_maes),  # Average of per-eval_id MAEs
                "num_valid_eval_ids": len(eval_id_maes),
                "num_unique_originals": len(all_original_values),
                "num_total_generated": len(all_generated_values),
                "original_values": all_original_values,
                "generated_values": all_generated_values,
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
    if similarity.get("similarities"):
        axes[1,0].hist(similarity["similarities"], bins=20, alpha=0.7, color="purple")
        axes[1,0].axvline(similarity["avg_similarity_per_molecule"], color="red", linestyle="--", 
                         label=f"Per-mol avg: {similarity['avg_similarity_per_molecule']:.3f}")
        axes[1,0].axvline(similarity["global_avg_similarity"], color="orange", linestyle=":", 
                         label=f"Global avg: {similarity['global_avg_similarity']:.3f}")
        axes[1,0].set_title("Tanimoto Similarity Distribution")
        axes[1,0].legend(fontsize=8)
    
    # Properties - show MAE for AtomicLogP as an example
    if "AtomicLogP" in properties:
        logp = properties["AtomicLogP"]
        axes[1,1].boxplot([logp["original_values"], logp["generated_values"]], 
                         tick_labels=["Original", "Generated"])
        axes[1,1].set_title(f"LogP (MAE: {logp['mae']:.3f})")
    elif properties:
        # If LogP not available, show the first available property
        first_prop = next(iter(properties.keys()))
        prop_data = properties[first_prop]
        axes[1,1].boxplot([prop_data["original_values"], prop_data["generated_values"]], 
                         tick_labels=["Original", "Generated"])
        axes[1,1].set_title(f"{first_prop} (MAE: {prop_data['mae']:.3f})")
    
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
        try:
            orig_mol = Chem.MolFromSmiles(row["original"])
            if orig_mol:
                img = Draw.MolToImage(orig_mol, size=(200, 200))
                axes[i,0].imshow(img)
                axes[i,0].set_title("Original", fontweight="bold")
                axes[i,0].axis('off')
        except Exception:
            axes[i,0].text(0.5, 0.5, "Invalid\nMolecule", ha='center', va='center', transform=axes[i,0].transAxes)
            axes[i,0].axis('off')
        
        # Generated molecules
        for j, (gen_smiles, sim) in enumerate(zip(row["generated"], row["similarities"])):
            if j < 5:
                try:
                    gen_mol = Chem.MolFromSmiles(gen_smiles)
                    if gen_mol:
                        img = Draw.MolToImage(gen_mol, size=(200, 200))
                        axes[i,j+1].imshow(img)
                        color = "green" if sim >= 0.7 else "orange" if sim >= 0.5 else "red"
                        axes[i,j+1].set_title(f"Sim: {sim:.3f}", color=color)
                        axes[i,j+1].axis('off')
                except Exception:
                    axes[i,j+1].text(0.5, 0.5, "Invalid\nMolecule", ha='center', va='center', transform=axes[i,j+1].transAxes)
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
        f.write("  Note: Exact matches counted once per unique original molecule\n\n")
        
        # Uniqueness
        u = results.uniqueness
        f.write(f"Global uniqueness: {u['unique_generated']}/{u['total_valid']} ({u['global_uniqueness']:.1%})\n")
        f.write(f"Avg per-sample uniqueness: {u['avg_per_sample_uniqueness']:.1%}\n\n")
        
        # Similarity
        s = results.similarity
        if s.get("similarities"):
            f.write(f"Tanimoto similarity (per-molecule avg): {s['avg_similarity_per_molecule']:.3f} ± {s['std_similarity_per_molecule']:.3f}\n")
            f.write(f"Tanimoto similarity (global avg): {s['global_avg_similarity']:.3f}\n")
            f.write(f"Number of molecules with similarities: {s['num_molecules_with_similarities']}\n\n")
        
        # Properties with MAE and benchmark comparison
        f.write("Molecular Property Analysis (MAE = Mean Absolute Error):\n")
        f.write("-" * 70 + "\n")
        
        # Benchmark values from comparison table
        benchmarks = {
            'AtomicLogP': {
                'merged_inchikey_DreaMS': 0.89,
                'unmerged_murcko_NIST': 1.548, 
                'unmerged_DiffMS_test': 0.76
            },
            'PolarSurfaceArea': {
                'merged_inchikey_DreaMS': 17.24,
                'unmerged_murcko_NIST': 27.7,
                'unmerged_DiffMS_test': 7.88
            },
            'FractionCSP3': {
                'merged_inchikey_DreaMS': 0.076,
                'unmerged_murcko_NIST': 0.156,
                'unmerged_DiffMS_test': 0.03
            },
            'QED': {
                'merged_inchikey_DreaMS': 0.078,
                'unmerged_murcko_NIST': 0.125,
                'unmerged_DiffMS_test': 0.10
            },
            'SyntheticAccessibility': {
                'merged_inchikey_DreaMS': 0.367,
                'unmerged_murcko_NIST': 0.692,
                'unmerged_DiffMS_test': 1.19
            },
            'BertzComplexity': {
                'merged_inchikey_DreaMS': 123.96,
                'unmerged_murcko_NIST': 177.679,
                'unmerged_DiffMS_test': 82.57
            },
            'NumHAcceptors': {
                'merged_inchikey_DreaMS': 0.87,
                'unmerged_murcko_NIST': 1.344,
                'unmerged_DiffMS_test': 0.32
            },
            'NumHDonors': {
                'merged_inchikey_DreaMS': 0.81,
                'unmerged_murcko_NIST': 1.291,
                'unmerged_DiffMS_test': 0.68
            },
            'NumRotatableBonds': {
                'merged_inchikey_DreaMS': 1.508,
                'unmerged_murcko_NIST': 3.472,
                'unmerged_DiffMS_test': 3.14
            },
            'NumAromaticRings': {
                'merged_inchikey_DreaMS': 0.296,
                'unmerged_murcko_NIST': 0.5512,
                'unmerged_DiffMS_test': 0.40
            },
            'NumAliphaticRings': {
                'merged_inchikey_DreaMS': 0.4651,
                'unmerged_murcko_NIST': 0.962,
                'unmerged_DiffMS_test': 1.44
            }
        }
        
        for prop, stats in results.properties.items():
            f.write(f"{prop}:\n")
            f.write(f"  Original: {stats['original_mean']:.3f} ± {stats['original_std']:.3f} (n={stats['num_unique_originals']})\n")
            f.write(f"  Generated: {stats['generated_mean']:.3f} ± {stats['generated_std']:.3f} (n={stats['num_total_generated']})\n")
            f.write(f"  MAE: {stats['mae']:.3f} (averaged over {stats['num_valid_eval_ids']} eval_ids)\n")
            f.write(f"  Valid eval_ids: {stats['num_valid_eval_ids']}\n")
            
            # Benchmark comparison
            if prop in benchmarks:
                f.write("  Benchmark Comparison:\n")
                for model_name, benchmark_mae in benchmarks[prop].items():
                    our_mae = stats['mae']
                    improvement = ((benchmark_mae - our_mae) / benchmark_mae * 100) if benchmark_mae != 0 else 0
                    status = "BETTER" if our_mae < benchmark_mae else "WORSE"
                    f.write(f"    vs {model_name}: {benchmark_mae:.3f} -> {our_mae:.3f} ({improvement:+.1f}% {status})\n")
            f.write("\n")
        
        # Overall benchmark summary
        f.write("=" * 70 + "\n")
        f.write("BENCHMARK SUMMARY - MAE Comparison\n")
        f.write("=" * 70 + "\n")
        
        # Calculate summary statistics for each benchmark model
        benchmark_models = ['merged_inchikey_DreaMS', 'unmerged_murcko_NIST', 'unmerged_DiffMS_test']
        
        for model_name in benchmark_models:
            f.write(f"\nComparison vs {model_name}:\n")
            f.write("-" * 50 + "\n")
            
            better_count = 0
            worse_count = 0
            total_improvements = []
            
            for prop, stats in results.properties.items():
                if prop in benchmarks and model_name in benchmarks[prop]:
                    benchmark_mae = benchmarks[prop][model_name]
                    our_mae = stats['mae']
                    improvement = ((benchmark_mae - our_mae) / benchmark_mae * 100) if benchmark_mae != 0 else 0
                    total_improvements.append(improvement)
                    
                    if our_mae < benchmark_mae:
                        better_count += 1
                        status_symbol = "✓"
                    else:
                        worse_count += 1
                        status_symbol = "✗"
                    
                    f.write(f"  {status_symbol} {prop:<20}: {improvement:+6.1f}%\n")
            
            if total_improvements:
                avg_improvement = np.mean(total_improvements)
                f.write(f"\nSummary: {better_count} better, {worse_count} worse\n")
                f.write(f"Average improvement: {avg_improvement:+.1f}%\n")


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
    logging.info(f"Avg similarity: {results.similarity.get('avg_similarity_per_molecule', 0.0):.3f}")
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