#!/usr/bin/env python3
"""Process MSG dataset: convert InChI to SMILES and generate fingerprints."""

import os
import polars as pl
from tqdm import tqdm

# Import the rdkit utilities from the md4 package
from md4 import rdkit_utils


def process_msg_dataset(input_file: str, output_file: str, fp_radius: int = 2, fp_bits: int = 2048):
    """
    Process MSG dataset from InChI format to SMILES with fingerprints.
    
    Args:
        input_file: Path to the input CSV file with InChI data
        output_file: Path to save the processed parquet file
        fp_radius: Radius for Morgan fingerprints (default: 2)
        fp_bits: Number of bits for fingerprints (default: 2048)
    """
    print(f"Loading MSG dataset from {input_file}...")
    
    # Read the CSV file with Polars
    df = pl.read_csv(input_file)
    print(f"Loaded {len(df)} InChI entries")
    
    # Convert InChI to SMILES and generate features
    processed_data = []
    failed_conversions = 0
    
    smiles_list = []
    fingerprint_list = []
    print("Converting InChI to SMILES and generating fingerprints...")
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Processing molecules"):
        inchi = row["inchi"]
        
        # Convert InChI to SMILES
        smiles = rdkit_utils.inchi_to_smiles(inchi)
        if smiles is None:
            failed_conversions += 1
            continue

            
        # Generate molecular features (fingerprints)
        features = rdkit_utils.get_molecule_features(
            smiles, radius=fp_radius, n_bits=fp_bits
        )
        if features is None:
            failed_conversions += 1
            continue
            
        smiles_list.append(smiles)
        fingerprint_list.append(features["fingerprint"])
    
    print(f"Successfully processed {len(smiles_list)} molecules")
    print(f"Failed to process {failed_conversions} molecules")
    
    # Create Polars dataframe

    processed_data = {
        "smiles": smiles_list,
        "fingerprint": fingerprint_list,
    }
    processed_df = pl.DataFrame(processed_data)

    print(processed_df.head())
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as parquet file
    processed_df.write_parquet(output_file)
    print(f"Saved processed dataset to {output_file}")
    
    # Print summary statistics
    print(f"\nDataset summary:")
    print(f"- Total processed molecules: {len(processed_df)}")
    print(f"- Fingerprint dimensions: {fp_bits}")
    print(f"- Morgan fingerprint radius: {fp_radius}")
    
    return processed_df


def main():
    """Main function to process the MSG dataset."""
    input_file = "./data/msg/msg_text.csv"
    output_file = "./data/msg/msg_processed.parquet"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Process the dataset
    processed_df = process_msg_dataset(
        input_file=input_file,
        output_file=output_file,
        fp_radius=2,
        fp_bits=2048
    )
    
    # Display sample of processed data
    print(f"\nSample of processed data:")
    print(processed_df.head())


if __name__ == "__main__":
    main()
