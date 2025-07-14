#!/usr/bin/env python3
"""Analysis script for MSG evaluation results.

This script can analyze results in two ways:
1. From a combined results file (CSV or parquet):
   python analyze_msg_results.py --results_file=./results/msg_eval_results.csv
   
2. From intermediate batch CSV files:
   python analyze_msg_results.py --intermediate_dir=./results/intermediate/

The script will automatically detect the format and create sample IDs if needed.

Optional filtering:
To filter out results with overlapping InChI key prefixes, provide:
   --overlapping_prefixes_file=./overlapps.csv

This will remove any results where the original SMILES has an InChI key prefix 
(first 14 characters) that appears in the overlapping prefixes CSV file.
"""

import os
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from absl import app, flags, logging
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, inchi
from rdkit.Chem import Draw

# Import rdkit utilities from md4

FLAGS = flags.FLAGS
flags.DEFINE_string("results_file", "./results/msg_eval_results.csv", "Path to evaluation results (CSV or parquet)")
flags.DEFINE_string("intermediate_dir", "", "Directory containing intermediate batch CSV files (alternative to results_file)")
flags.DEFINE_string("output_dir", "./results/analysis/", "Directory to save analysis results")
flags.DEFINE_string("overlapping_prefixes_file", "", "Path to CSV file containing overlapping InChI key prefixes to filter out")
flags.DEFINE_bool("plot_figures", True, "Whether to generate plots")
flags.DEFINE_bool("molecular_grid", True, "Whether to generate molecular structure grid")


def load_overlapping_prefixes(file_path: str) -> Set[str]:
    """Load overlapping InChI key prefixes from CSV file."""
    if not file_path or not os.path.exists(file_path):
        logging.info("No overlapping prefixes file provided or file doesn't exist. Skipping filtering.")
        return set()
    
    try:
        df = pl.read_csv(file_path)
        prefixes = set(df.select("overlapping_prefix").to_series().to_list())
        logging.info(f"Loaded {len(prefixes)} overlapping InChI key prefixes from {file_path}")
        return prefixes
    except Exception as e:
        logging.warning(f"Failed to load overlapping prefixes from {file_path}: {e}")
        return set()


def get_inchi_key_prefix(smiles: str) -> str:
    """Get the first 14 characters of InChI key from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            inchi_key = inchi.MolToInchiKey(mol)
            return inchi_key[:14] if inchi_key else ""
        return ""
    except Exception:
        return ""


def filter_overlapping_samples(df: pl.DataFrame, overlapping_prefixes: Set[str]) -> pl.DataFrame:
    """Filter out samples with overlapping InChI key prefixes."""
    if not overlapping_prefixes:
        logging.info("No overlapping prefixes to filter, returning original data")
        return df
    
    # Add InChI key prefix column
    df = df.with_columns(
        pl.col("original_smiles").map_elements(get_inchi_key_prefix, return_dtype=pl.String).alias("inchi_key_prefix")
    )
    
    # Count original samples
    original_count = len(df)
    original_unique_samples = df.select("original_smiles").unique().height
    
    # Filter out overlapping samples
    df_filtered = df.filter(~pl.col("inchi_key_prefix").is_in(list(overlapping_prefixes)))
    
    # Count after filtering
    filtered_count = len(df_filtered)
    filtered_unique_samples = df_filtered.select("original_smiles").unique().height
    
    removed_count = original_count - filtered_count
    removed_unique_samples = original_unique_samples - filtered_unique_samples
    
    logging.info(f"Filtering results:")
    logging.info(f"- Original: {original_count} total samples from {original_unique_samples} unique molecules")
    logging.info(f"- Removed: {removed_count} samples from {removed_unique_samples} overlapping molecules")
    logging.info(f"- Remaining: {filtered_count} samples from {filtered_unique_samples} unique molecules")
    
    # Drop the temporary inchi_key_prefix column
    df_filtered = df_filtered.drop("inchi_key_prefix")
    
    return df_filtered


def load_results(results_file: str) -> pl.DataFrame:
    """Load evaluation results from CSV or parquet format."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Determine file format and load accordingly
    if results_file.endswith('.parquet'):
        df = pl.read_parquet(results_file)
    elif results_file.endswith('.csv'):
        df = pl.read_csv(results_file)
    else:
        # Try to detect format
        try:
            df = pl.read_parquet(results_file)
        except:
            df = pl.read_csv(results_file)
    
    logging.info(f"Loaded {len(df)} generated samples")
    
    # Clean generated SMILES by removing spaces (common issue with tokenized output)
    df = df.with_columns(
        pl.col("generated_smiles").str.replace_all(" ", "").alias("generated_smiles")
    )
    logging.info("Cleaned spaces from generated SMILES strings")
    
    # Add canonical SMILES for proper comparison
    def canonicalize_smiles(smiles):
        """Canonicalize SMILES string, return None if invalid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.CanonSmiles(smiles)
            return None
        except:
            return None
    
    df = df.with_columns([
        pl.col("original_smiles").map_elements(canonicalize_smiles, return_dtype=pl.String).alias("original_canonical"),
        pl.col("generated_smiles").map_elements(canonicalize_smiles, return_dtype=pl.String).alias("generated_canonical")
    ])
    logging.info("Added canonical SMILES columns for proper comparison")
    
    # Check if we have sample_id column, if not create it
    if "sample_id" not in df.columns:
        logging.info("No sample_id column found, creating from original_smiles groups")
        
        # Create sample IDs by grouping original SMILES
        # First, get unique original SMILES and assign IDs
        unique_originals = df.select("original_smiles").unique().with_row_index("sample_id")
        
        # Join back to add sample_id to all rows
        df = df.join(unique_originals, on="original_smiles", how="left")
        
        # Add generation_idx within each sample group if not present
        if "generation_idx" not in df.columns:
            df = df.with_columns(
                pl.col("original_smiles").rank(method="ordinal").over("original_smiles").sub(1).alias("generation_idx")
            )
    
    # Count unique samples
    unique_originals = df.select("sample_id").unique().height
    logging.info(f"From {unique_originals} original molecules")
    
    return df


def load_intermediate_results(intermediate_dir: str) -> pl.DataFrame:
    """Load and combine intermediate batch results from CSV files."""
    if not os.path.exists(intermediate_dir):
        raise FileNotFoundError(f"Intermediate directory not found: {intermediate_dir}")
    
    # Find all batch CSV files
    batch_files = sorted([
        f for f in os.listdir(intermediate_dir) 
        if f.startswith("batch_") and f.endswith(".csv")
    ])
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {intermediate_dir}")
    
    logging.info(f"Found {len(batch_files)} batch files in {intermediate_dir}")
    
    # Read and combine all batch files
    all_dfs = []
    for batch_file in batch_files:
        batch_path = os.path.join(intermediate_dir, batch_file)
        batch_df = pl.read_csv(batch_path)
        all_dfs.append(batch_df)
    
    # Combine all DataFrames
    df = pl.concat(all_dfs)
    logging.info(f"Combined {len(df)} generated samples from batch files")
    
    # Clean generated SMILES by removing spaces (common issue with tokenized output)
    df = df.with_columns(
        pl.col("generated_smiles").str.replace_all(" ", "").alias("generated_smiles")
    )
    logging.info("Cleaned spaces from generated SMILES strings")
    
    # Add canonical SMILES for proper comparison
    def canonicalize_smiles(smiles):
        """Canonicalize SMILES string, return None if invalid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.CanonSmiles(smiles)
            return None
        except:
            return None
    
    df = df.with_columns([
        pl.col("original_smiles").map_elements(canonicalize_smiles, return_dtype=pl.String).alias("original_canonical"),
        pl.col("generated_smiles").map_elements(canonicalize_smiles, return_dtype=pl.String).alias("generated_canonical")
    ])
    logging.info("Added canonical SMILES columns for proper comparison")
    
    return df


def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """Calculate molecular properties from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        }
    except Exception:
        return None


def analyze_validity(df: pl.DataFrame) -> Dict[str, float]:
    """Analyze validity of generated SMILES."""
    # Count valid molecules using canonical SMILES (None means invalid)
    valid_generated = df.filter(pl.col("generated_canonical").is_not_null())
    valid_count = len(valid_generated)
    total_count = len(df)
    
    validity_rate = valid_count / total_count if total_count > 0 else 0
    
    # Count exact matches (canonical SMILES are identical)
    exact_matches = df.filter(
        (pl.col("original_canonical").is_not_null()) & 
        (pl.col("generated_canonical").is_not_null()) &
        (pl.col("original_canonical") == pl.col("generated_canonical"))
    )
    exact_match_count = len(exact_matches)
    exact_match_rate = exact_match_count / total_count if total_count > 0 else 0
    
    logging.info(f"Validity Analysis:")
    logging.info(f"- Valid SMILES: {valid_count}/{total_count} ({validity_rate:.2%})")
    logging.info(f"- Exact matches: {exact_match_count}/{total_count} ({exact_match_rate:.2%})")
    
    return {
        "validity_rate": validity_rate, 
        "valid_count": valid_count, 
        "total_count": total_count,
        "exact_match_rate": exact_match_rate,
        "exact_match_count": exact_match_count
    }


def analyze_uniqueness(df: pl.DataFrame) -> Dict[str, float]:
    """Analyze uniqueness of generated SMILES using canonical forms."""
    # Filter for valid generated molecules only
    valid_df = df.filter(pl.col("generated_canonical").is_not_null())
    
    # Global uniqueness using canonical SMILES
    unique_canonical = valid_df.select("generated_canonical").unique().height
    total_valid = len(valid_df)
    global_uniqueness = unique_canonical / total_valid if total_valid > 0 else 0
    
    # Per-sample uniqueness (within each set generated for same original)
    per_sample_uniqueness = []
    
    for sample_id in df.select("sample_id").unique().to_series():
        sample_df = valid_df.filter(pl.col("sample_id") == sample_id)
        if len(sample_df) > 0:
            unique_in_sample = sample_df.select("generated_canonical").unique().height
            total_in_sample = len(sample_df)
            uniqueness = unique_in_sample / total_in_sample if total_in_sample > 0 else 0
            per_sample_uniqueness.append(uniqueness)
    
    avg_per_sample_uniqueness = np.mean(per_sample_uniqueness) if per_sample_uniqueness else 0
    
    logging.info(f"Uniqueness Analysis (using canonical SMILES):")
    logging.info(f"- Global uniqueness: {unique_canonical}/{total_valid} ({global_uniqueness:.2%})")
    logging.info(f"- Average per-sample uniqueness: {avg_per_sample_uniqueness:.2%}")
    
    return {
        "global_uniqueness": global_uniqueness,
        "avg_per_sample_uniqueness": avg_per_sample_uniqueness,
        "unique_generated": unique_canonical,
        "total_valid": total_valid,
    }


def analyze_similarity_to_original(df: pl.DataFrame) -> Dict[str, float]:
    """Analyze similarity between generated and original SMILES."""
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    
    similarities = []
    
    fpgen = AllChem.GetRDKitFPGenerator()
    for row in df.iter_rows(named=True):
        original_smiles = row["original_smiles"]
        generated_smiles = row["generated_smiles"]
        
        try:
            original_mol = Chem.MolFromSmiles(original_smiles)
            generated_mol = Chem.MolFromSmiles(generated_smiles)

            if original_mol is not None and generated_mol is not None:
                # Generate fingerprints from molecule object
                fp0 = fpgen.GetFingerprint(original_mol)
                fp1 = fpgen.GetFingerprint(generated_mol)
                
                # Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(fp0, fp1)
                similarities.append(similarity)
        except:
            continue
    
    if similarities:
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        logging.info(f"Similarity Analysis:")
        logging.info(f"- Average Tanimoto similarity: {avg_similarity:.3f} ± {std_similarity:.3f}")
        logging.info(f"- Max similarity: {np.max(similarities):.3f}")
        logging.info(f"- Min similarity: {np.min(similarities):.3f}")
        
        return {
            "avg_similarity": avg_similarity,
            "std_similarity": std_similarity,
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "similarities": similarities,
        }
    else:
        logging.warning("No valid similarity calculations could be performed")
        return {
            "avg_similarity": 0.0, 
            "std_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
            "similarities": []
        }


def analyze_molecular_properties(df: pl.DataFrame) -> Dict[str, Dict]:
    """Analyze molecular properties of generated vs original molecules."""
    # Filter for valid molecules only (those with canonical SMILES)
    valid_df = df.filter(
        (pl.col("original_canonical").is_not_null()) & 
        (pl.col("generated_canonical").is_not_null())
    )
    
    if len(valid_df) == 0:
        logging.warning("No valid molecules found for property analysis")
        return {}
    
    original_props = []
    generated_props = []
    
    # Get unique original molecules using canonical SMILES
    unique_originals = valid_df.group_by("original_canonical").first()
    
    for row in unique_originals.iter_rows(named=True):
        original_smiles = row["original_smiles"]
        props = calculate_molecular_properties(original_smiles)
        if props:
            original_props.append(props)
    
    # Get all valid generated molecules
    for row in valid_df.iter_rows(named=True):
        generated_smiles = row["generated_smiles"]
        props = calculate_molecular_properties(generated_smiles)
        if props:
            generated_props.append(props)
    
    # Calculate statistics for each property
    results = {}
    prop_names = ["molecular_weight", "logp", "tpsa", "num_atoms", "num_bonds", "num_rings"]
    
    for prop in prop_names:
        if original_props and generated_props:
            orig_values = [p[prop] for p in original_props if prop in p]
            gen_values = [p[prop] for p in generated_props if prop in p]
            
            if orig_values and gen_values:
                results[prop] = {
                    "original_mean": np.mean(orig_values),
                    "original_std": np.std(orig_values),
                    "generated_mean": np.mean(gen_values),
                    "generated_std": np.std(gen_values),
                    "original_values": orig_values,
                    "generated_values": gen_values,
                }
                
                logging.info(f"{prop.replace('_', ' ').title()}:")
                logging.info(f"  Original:  {np.mean(orig_values):.2f} ± {np.std(orig_values):.2f}")
                logging.info(f"  Generated: {np.mean(gen_values):.2f} ± {np.std(gen_values):.2f}")
    
    return results


def create_molecular_grid(df: pl.DataFrame, output_dir: str, max_samples: int = 10, max_generations: int = 5):
    """Create a molecular grid showing original vs generated molecules ordered by similarity."""
    try:
        import matplotlib.pyplot as plt
        from rdkit.Chem import Draw
        from rdkit import DataStructs
        from io import BytesIO
        from PIL import Image
        
        # Filter for valid SMILES only
        def is_valid_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        
        def calculate_tanimoto_similarity(smiles1, smiles2):
            """Calculate Tanimoto similarity between two SMILES."""
            try:
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol1 is not None and mol2 is not None:
                    fpgen = AllChem.GetRDKitFPGenerator()
                    fp1 = fpgen.GetFingerprint(mol1)
                    fp2 = fpgen.GetFingerprint(mol2)
                    return DataStructs.TanimotoSimilarity(fp1, fp2)
                return 0.0
            except:
                return 0.0
        
        # Add validity columns
        df_valid = df.with_columns([
            pl.col("original_smiles").map_elements(is_valid_smiles, return_dtype=pl.Boolean).alias("original_valid"),
            pl.col("generated_smiles").map_elements(is_valid_smiles, return_dtype=pl.Boolean).alias("generated_valid")
        ])
        
        # Get samples with valid original SMILES
        valid_samples = df_valid.filter(pl.col("original_valid") == True).group_by("sample_id").first()
        
        if len(valid_samples) == 0:
            logging.warning("No valid original SMILES found for molecular grid")
            return
        
        # Create the grid
        fig, axes = plt.subplots(max_samples, max_generations + 1, figsize=(18, 3 * max_samples))
        if max_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("Molecular Comparison: Original vs Generated (sorted by Tanimoto similarity)", fontsize=14)
        
        # Fill the grid with valid molecules
        filled_rows = 0
        for sample_idx, sample_row in enumerate(valid_samples.iter_rows(named=True)):
            if filled_rows >= max_samples:
                break
                
            sample_id = sample_row["sample_id"]
            original_smiles = sample_row["original_smiles"]
            
            # Get valid generated molecules for this sample with similarity scores
            sample_generated = df_valid.filter(
                (pl.col("sample_id") == sample_id) & 
                (pl.col("generated_valid") == True)
            )
            
            # Calculate similarities and sort by similarity (highest first)
            molecules_with_similarity = []
            for gen_row in sample_generated.iter_rows(named=True):
                gen_smiles = gen_row["generated_smiles"]
                similarity = calculate_tanimoto_similarity(original_smiles, gen_smiles)
                molecules_with_similarity.append((gen_smiles, similarity))
            
            # Sort by similarity (highest first)
            molecules_with_similarity.sort(key=lambda x: x[1], reverse=True)
            
            # Draw original molecule
            original_mol = Chem.MolFromSmiles(original_smiles)
            img = Draw.MolToImage(original_mol, size=(300, 300))
            axes[filled_rows, 0].imshow(img)
            axes[filled_rows, 0].set_title(f"Original\n{original_smiles[:25]}...", fontsize=9, weight='bold')
            axes[filled_rows, 0].axis('off')
            
            # Draw generated molecules sorted by similarity
            for gen_idx, (gen_smiles, similarity) in enumerate(molecules_with_similarity[:max_generations]):
                plot_col = gen_idx + 1
                gen_mol = Chem.MolFromSmiles(gen_smiles)
                
                # Create image with colored border based on similarity
                img = Draw.MolToImage(gen_mol, size=(300, 300))
                axes[filled_rows, plot_col].imshow(img)
                
                # Color-code title based on similarity
                if similarity >= 0.7:
                    color = 'green'
                elif similarity >= 0.5:
                    color = 'orange' 
                elif similarity >= 0.3:
                    color = 'red'
                else:
                    color = 'darkred'
                
                axes[filled_rows, plot_col].set_title(
                    f"Generated {gen_idx+1}\nSimilarity: {similarity:.3f}\n{gen_smiles[:20]}...", 
                    fontsize=8, color=color, weight='bold'
                )
                axes[filled_rows, plot_col].axis('off')
                
                # Add colored border around the subplot
                for spine in axes[filled_rows, plot_col].spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(3)
            
            # Fill remaining columns if we have fewer than max_generations
            for col_idx in range(len(molecules_with_similarity), max_generations):
                plot_col = col_idx + 1
                axes[filled_rows, plot_col].text(0.5, 0.5, "No Valid\nGeneration", 
                                               ha='center', va='center', 
                                               transform=axes[filled_rows, plot_col].transAxes)
                axes[filled_rows, plot_col].set_title(f"Generated {col_idx+1}\n(None)", fontsize=8)
                axes[filled_rows, plot_col].axis('off')
            
            filled_rows += 1
        
        # Hide any unused rows
        for row_idx in range(filled_rows, max_samples):
            for col_idx in range(max_generations + 1):
                axes[row_idx, col_idx].axis('off')
        
        # Add legend for similarity color coding
        legend_text = "Similarity Color Code:\nGreen: ≥0.7 (High)\nOrange: 0.5-0.7 (Medium)\nRed: 0.3-0.5 (Low)\nDark Red: <0.3 (Very Low)"
        fig.text(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        grid_path = os.path.join(output_dir, "molecular_grid_similarity.png")
        plt.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logging.info(f"Molecular similarity grid saved to {grid_path} (showing {filled_rows} rows of valid molecules)")
        
    except ImportError as e:
        logging.warning(f"Could not create molecular grid: missing dependency {e}")
    except Exception as e:
        logging.warning(f"Could not create molecular grid: {e}")


def create_plots(
    validity_stats: Dict,
    uniqueness_stats: Dict,
    similarity_stats: Dict,
    property_stats: Dict,
    output_dir: str,
):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            # Use default style if seaborn is not available
            pass
    
    # 1. Summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MSG Evaluation Summary", fontsize=16)
    
    # Validity and Exact Matches
    categories = ["Valid", "Invalid", "Exact Match"]
    counts = [
        validity_stats["valid_count"], 
        validity_stats["total_count"] - validity_stats["valid_count"],
        validity_stats["exact_match_count"]
    ]
    colors = ["green", "red", "gold"]
    
    axes[0, 0].bar(categories, counts, color=colors)
    axes[0, 0].set_title(f"Validity: {validity_stats['validity_rate']:.1%} | Exact Match: {validity_stats['exact_match_rate']:.1%}")
    axes[0, 0].set_ylabel("Count")
    
    # Uniqueness
    axes[0, 1].bar(["Unique", "Duplicate"], 
                   [uniqueness_stats["unique_generated"], 
                    uniqueness_stats["total_valid"] - uniqueness_stats["unique_generated"]], 
                   color=["blue", "orange"])
    axes[0, 1].set_title(f"Global Uniqueness: {uniqueness_stats['global_uniqueness']:.1%}")
    axes[0, 1].set_ylabel("Count")
    
    # Similarity distribution
    if "similarities" in similarity_stats and similarity_stats["similarities"]:
        axes[1, 0].hist(similarity_stats["similarities"], bins=20, alpha=0.7, color="purple")
        axes[1, 0].set_title(f"Tanimoto Similarity Distribution")
        axes[1, 0].set_xlabel("Similarity")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].axvline(similarity_stats["avg_similarity"], color="red", linestyle="--", 
                          label=f"Mean: {similarity_stats['avg_similarity']:.3f}")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No valid similarity\ncalculations", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Tanimoto Similarity Distribution")
        axes[1, 0].set_xlabel("Similarity")
        axes[1, 0].set_ylabel("Frequency")
    
    # Property comparison (molecular weight as example)
    if "molecular_weight" in property_stats:
        mw_stats = property_stats["molecular_weight"]
        axes[1, 1].boxplot([mw_stats["original_values"], mw_stats["generated_values"]], 
                          tick_labels=["Original", "Generated"])
        axes[1, 1].set_title("Molecular Weight Distribution")
        axes[1, 1].set_ylabel("Molecular Weight (Da)")
    else:
        axes[1, 1].text(0.5, 0.5, "No property\ndata available", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Molecular Weight Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_stats.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Property comparison plots
    if property_stats:
        n_props = len(property_stats)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (prop, stats) in enumerate(property_stats.items()):
            if i < len(axes):
                axes[i].boxplot([stats["original_values"], stats["generated_values"]], 
                               tick_labels=["Original", "Generated"])
                axes[i].set_title(prop.replace('_', ' ').title())
                axes[i].set_ylabel(prop.replace('_', ' ').title())
        
        # Hide unused subplots
        for i in range(len(property_stats), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "property_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    logging.info(f"Plots saved to {output_dir}")


def generate_report(
    validity_stats: Dict,
    uniqueness_stats: Dict,
    similarity_stats: Dict,
    property_stats: Dict,
    output_dir: str,
):
    """Generate a summary report."""
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, "w") as f:
        f.write("MSG Evaluation Report\n")
        f.write("====================\n\n")
        
        f.write("1. Validity Analysis\n")
        f.write(f"   - Total generated samples: {validity_stats['total_count']}\n")
        f.write(f"   - Valid SMILES: {validity_stats['valid_count']}\n")
        f.write(f"   - Validity rate: {validity_stats['validity_rate']:.2%}\n")
        f.write(f"   - Exact matches: {validity_stats['exact_match_count']}/{validity_stats['total_count']} ({validity_stats['exact_match_rate']:.2%})\n\n")
        
        f.write("2. Uniqueness Analysis\n")
        f.write(f"   - Unique molecules (global): {uniqueness_stats['unique_generated']}\n")
        f.write(f"   - Global uniqueness rate: {uniqueness_stats['global_uniqueness']:.2%}\n")
        f.write(f"   - Average per-sample uniqueness: {uniqueness_stats['avg_per_sample_uniqueness']:.2%}\n\n")
        
        f.write("3. Similarity Analysis\n")
        if "avg_similarity" in similarity_stats and similarity_stats["avg_similarity"] > 0:
            f.write(f"   - Average Tanimoto similarity: {similarity_stats['avg_similarity']:.3f}\n")
            f.write(f"   - Similarity std: {similarity_stats['std_similarity']:.3f}\n")
            if "max_similarity" in similarity_stats:
                f.write(f"   - Max similarity: {similarity_stats['max_similarity']:.3f}\n")
            if "min_similarity" in similarity_stats:
                f.write(f"   - Min similarity: {similarity_stats['min_similarity']:.3f}\n")
        else:
            f.write("   - No valid similarity calculations could be performed\n")
        f.write("\n")
        
        f.write("4. Molecular Property Analysis\n")
        for prop, stats in property_stats.items():
            f.write(f"   {prop.replace('_', ' ').title()}:\n")
            f.write(f"     - Original:  {stats['original_mean']:.2f} ± {stats['original_std']:.2f}\n")
            f.write(f"     - Generated: {stats['generated_mean']:.2f} ± {stats['generated_std']:.2f}\n")
    
    logging.info(f"Report saved to {report_path}")


def main(argv):
    del argv  # Unused
    
    logging.info("Starting MSG results analysis...")
    
    # Load overlapping prefixes if provided
    overlapping_prefixes = load_overlapping_prefixes(FLAGS.overlapping_prefixes_file)
    
    # Load results from either results_file or intermediate_dir
    if FLAGS.intermediate_dir:
        logging.info(f"Loading from intermediate directory: {FLAGS.intermediate_dir}")
        df = load_intermediate_results(FLAGS.intermediate_dir)
        
        # Filter out overlapping samples
        df = filter_overlapping_samples(df, overlapping_prefixes)
        
        # Process the combined intermediate results to add sample IDs
        if "sample_id" not in df.columns:
            logging.info("Creating sample IDs from original_smiles groups")
            unique_originals = df.select("original_smiles").unique().with_row_index("sample_id")
            df = df.join(unique_originals, on="original_smiles", how="left")
            
            if "generation_idx" not in df.columns:
                df = df.with_columns(
                    pl.col("original_smiles").rank(method="ordinal").over("original_smiles").sub(1).alias("generation_idx")
                )
    else:
        logging.info(f"Loading from results file: {FLAGS.results_file}")
        df = load_results(FLAGS.results_file)
        
        # Filter out overlapping samples
        df = filter_overlapping_samples(df, overlapping_prefixes)
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Run analyses
    validity_stats = analyze_validity(df)
    uniqueness_stats = analyze_uniqueness(df)
    similarity_stats = analyze_similarity_to_original(df)
    property_stats = analyze_molecular_properties(df)
    
    # Generate visualizations
    if FLAGS.plot_figures:
        try:
            create_plots(validity_stats, uniqueness_stats, similarity_stats, 
                        property_stats, FLAGS.output_dir)
        except Exception as e:
            logging.warning(f"Could not generate plots: {e}")
    
    # Generate molecular grid
    if FLAGS.molecular_grid:
        try:
            create_molecular_grid(df, FLAGS.output_dir)
        except Exception as e:
            logging.warning(f"Could not generate molecular grid: {e}")
    
    # Generate report
    generate_report(validity_stats, uniqueness_stats, similarity_stats, 
                   property_stats, FLAGS.output_dir)
    
    logging.info("Analysis completed successfully!")


if __name__ == "__main__":
    app.run(main) 