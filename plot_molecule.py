#!/usr/bin/env python3
"""
Script to parse SMILES strings and plot molecular structures using RDKit.
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def parse_smiles(smiles_string):
    """Parse a SMILES string and return the molecule object."""
    try:
        # Clean up the SMILES string (remove extra spaces)
        smiles_clean = ''.join(smiles_string.split())
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles_clean)
        
        if mol is None:
            print(f"Error: Could not parse SMILES string: {smiles_string}")
            return None
            
        # Add hydrogens for better visualization
        mol = Chem.AddHs(mol)
        
        # Generate 2D coordinates
        rdDepictor.Compute2DCoords(mol)
        
        return mol
        
    except Exception as e:
        print(f"Error parsing SMILES: {e}")
        return None

def plot_molecule(mol, title="Molecular Structure", save_path=None):
    """Plot the molecular structure using RDKit."""
    if mol is None:
        print("No molecule to plot")
        return
    
    try:
        # Create a drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        
        # Draw the molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get the drawing as PNG
        png_data = drawer.GetDrawingText()
        
        # Save or display
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(png_data)
            print(f"Molecule structure saved to: {save_path}")
        else:
            # Save to temporary file and display
            temp_path = "temp_molecule.png"
            with open(temp_path, 'wb') as f:
                f.write(png_data)
            
            # Display using matplotlib
            img = plt.imread(temp_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(title, fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"Error plotting molecule: {e}")

def get_molecule_info(mol):
    """Get basic information about the molecule."""
    if mol is None:
        return
    
    try:
        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        # Get molecular weight
        mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        
        # Get number of atoms
        num_atoms = mol.GetNumAtoms()
        
        # Get number of bonds
        num_bonds = mol.GetNumBonds()
        
        # Get canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        print(f"\nMolecule Information:")
        print(f"  Formula: {formula}")
        print(f"  Molecular Weight: {mol_weight:.2f}")
        print(f"  Number of Atoms: {num_atoms}")
        print(f"  Number of Bonds: {num_bonds}")
        print(f"  Canonical SMILES: {canonical_smiles}")
        
    except Exception as e:
        print(f"Error getting molecule info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Parse SMILES and plot molecular structures")
    parser.add_argument("smiles", nargs="?", help="SMILES string to parse")
    parser.add_argument("-f", "--file", help="Read SMILES from file")
    parser.add_argument("-o", "--output", help="Save plot to file instead of displaying")
    parser.add_argument("-i", "--info", action="store_true", help="Show molecule information")
    
    args = parser.parse_args()
    
    # Get SMILES string
    if args.smiles:
        smiles_string = args.smiles
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                smiles_string = f.read().strip()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return
    else:
        # Interactive mode
        smiles_string = input("Enter SMILES string: ")
    
    if not smiles_string:
        print("No SMILES string provided")
        return
    
    print(f"Parsing SMILES: {smiles_string}")
    
    # Parse the SMILES
    mol = parse_smiles(smiles_string)
    
    if mol is None:
        return
    
    # Show molecule information if requested
    if args.info:
        get_molecule_info(mol)
    
    # Plot the molecule
    title = f"Molecular Structure"
    plot_molecule(mol, title=title, save_path=args.output)

if __name__ == "__main__":
    main() 