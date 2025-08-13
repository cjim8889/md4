#!/usr/bin/env python3
"""
Script to train a SentencePiece tokenizer on SMILES strings and molecular formulas.
Data format: molecular_formula[SEP]smiles
Includes [SEP] as a special token for separating molecular formulas and SMILES.
"""

import os
import tempfile
import argparse
import logging
from typing import Tuple, Union

import tensorflow as tf
import datasets
import polars as pl

from sentencepiece import SentencePieceTrainer


def _dump_hf_data_to_textfile(
    dataset, maxchars: Union[int, None] = int(1e7), sep_token: str = "[SEP]", 
    smiles_column: str = "smiles", formula_column: str = "molecular_formula"
) -> Tuple[str, int]:
    """Write HuggingFace dataset to lines in a text file with formula[SEP]smiles format.
    Args:
      dataset: HuggingFace dataset or Polars DataFrame containing SMILES and molecular formula data.
      maxchars: int: approximate number of characters to save from dataset. If None or -1, process all data.
      sep_token: str: separator token to use between formula and SMILES.
      smiles_column: str: name of the SMILES column in the dataset.
      formula_column: str: name of the molecular formula column in the dataset.
    Returns:
      name of temp file with dataset text, exact number of characters dumped.
    """
    # Convert HuggingFace dataset to proper Polars DataFrame
    if hasattr(dataset, 'to_polars'):
        # HuggingFace dataset with polars format
        df = dataset.to_polars()
    elif hasattr(dataset, 'to_pandas'):
        # HuggingFace dataset without polars format - convert via pandas
        df = pl.from_pandas(dataset.to_pandas())
    else:
        # Assume it's already a Polars DataFrame
        df = dataset
    
    # Filter out rows with missing or invalid data using Polars operations
    df_clean = df.filter(
        (pl.col(smiles_column).is_not_null()) &
        (pl.col(formula_column).is_not_null()) &
        (pl.col(smiles_column).str.strip_chars() != "") &
        (pl.col(formula_column).str.strip_chars() != "") &
        (~pl.col(formula_column).str.to_lowercase().is_in(["none", "null", ""]))
    )
    
    # Create the formatted text using Polars string operations
    df_formatted = df_clean.with_columns([
        (pl.col(formula_column).str.strip_chars() + 
         pl.lit(sep_token) + 
         pl.col(smiles_column).str.strip_chars() + 
         pl.lit("\n")).alias("formatted_line")
    ])
    
    # Handle maxchars limit if specified
    if maxchars is not None and maxchars != -1:
        # Calculate cumulative character count
        df_with_len = df_formatted.with_columns([
            pl.col("formatted_line").str.len_chars().alias("line_length")
        ]).with_columns([
            pl.col("line_length").cum_sum().alias("cumulative_chars")
        ])
        
        # Filter to stay within maxchars limit
        df_limited = df_with_len.filter(pl.col("cumulative_chars") <= maxchars)
        lines_series = df_limited.select("formatted_line").to_series()
        char_count = int(df_limited.select("cumulative_chars").max().item()) if len(df_limited) > 0 else 0
    else:
        lines_series = df_formatted.select("formatted_line").to_series()
        char_count = int(df_formatted.select(pl.col("formatted_line").str.len_chars().sum()).item())
    
    # Write to temporary file
    temp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=os.path.join(temp_dir, "ds_chars"), mode='w', encoding='utf-8'
    ) as outfp:
        # Write all lines at once using Polars' efficient string operations
        outfp.write(''.join(lines_series.to_list()))
            
    return outfp.name, char_count


def _train_sentencepiece(
    dataset,
    *,
    vocab_size: int,
    maxchars: Union[int, None] = int(1e7),
    model_path: str,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    sep_token: str = "[SEP]",
    smiles_column: str = "smiles",
    formula_column: str = "molecular_formula",
):
    """Train SentencePiece tokenizer from HuggingFace dataset.
    Args:
      dataset: HuggingFace dataset
      vocab_size: int: size of vocab tokens to train.
      maxchars: int: number of characters to use for sentencepiece training. If None or -1, use all data.
      model_path: str: path of model file to save vocab model to.
      model_type: str: type of sentencepiece vocab to train.
      character_coverage: amount of characters covered by the model, good defaults
        are 0.9995 for languages with rich character set like Japanese or Chinese
        and 1.0 for other languages with small character set.
      sep_token: str: separator token to include as special token.
      smiles_column: str: name of the SMILES column in the dataset.
      formula_column: str: name of the molecular formula column in the dataset.
    Returns:
      path to the trained sentencepiece vocabulary model.
    """
    if model_path.startswith("gs://"):
        abs_model_path = model_path
    else:
        abs_model_path = os.path.abspath(os.path.expanduser(model_path))
    
    fname, actual_chars = _dump_hf_data_to_textfile(
        dataset, maxchars=maxchars, sep_token=sep_token,
        smiles_column=smiles_column, formula_column=formula_column
    )
    if maxchars is None or maxchars == -1:
        logging.info(f"Dumped all {actual_chars} characters to {fname}")
    else:
        logging.info(f"Dumped {actual_chars} characters (max {maxchars}) to {fname}")
    
    temp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=os.path.join(temp_dir, "sp_tmp")
    ) as model_fp:
        pass  # we just want a prefix'd tmp-filename
    
    # Include [SEP] as a special token
    user_defined_symbols = [sep_token]
    
    argstr = " ".join(
        [
            f"--input={fname}",
            f"--vocab_size={vocab_size}",
            f"--character_coverage={character_coverage}",
            f"--model_prefix={model_fp.name}",
            f"--model_type={model_type}",
            f"--user_defined_symbols={','.join(user_defined_symbols)}",
            "--input_sentence_size=80000000",
            "--add_dummy_prefix=false",
            "--num_threads=128",
            "--unk_piece=[UNK]",
            "--bos_piece=[BEGIN]",
            "--eos_piece=[END]",
            "--pad_piece=[PAD]",
            "--train_extremely_large_corpus=true",
            "--shuffle_input_sentence=true",
        ]
    )
    
    logging.info(f"Training SentencePiece with args: {argstr}")
    SentencePieceTrainer.Train(argstr)
    
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.
    copy_rename_path = abs_model_path + ".rntmp"
    tf.io.gfile.makedirs(os.path.dirname(abs_model_path))
    tf.io.gfile.copy(model_fp.name + ".model", copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
    logging.info("copied %s to %s", model_fp.name + ".model", abs_model_path)
    
    # Clean up temp files
    try:
        os.unlink(fname)
        os.unlink(model_fp.name + ".model")
        os.unlink(model_fp.name + ".vocab")
    except FileNotFoundError:
        pass
        
    return abs_model_path


def train_tokenizer(
    dataset_name: str,
    *,
    vocab_path: str,
    vocab_size: int,
    max_corpus_chars: Union[int, None],
    model_type: str = "unigram",
    sep_token: str = "[SEP]",
    smiles_column: str = "smiles",
    formula_column: str = "molecular_formula",
):
    """tokenizer training function
    Args:
        max_corpus_chars: int: maximum number of characters to use from corpus. If None or -1, use all data.
        model_type: str: type of sentencepiece model ('unigram', 'bpe', 'word', 'char').
    """
    logging.info("Loading dataset from HuggingFace...")
    ds = datasets.load_dataset(dataset_name, split="train").with_format("polars")
    logging.info("Dataset loaded successfully")
    
    logging.info("SentencePiece vocab not found, building one from data.")
    vocab_path = _train_sentencepiece(
        ds,
        vocab_size=vocab_size,
        maxchars=max_corpus_chars,
        model_path=vocab_path,
        model_type=model_type,
        sep_token=sep_token,
        smiles_column=smiles_column,
        formula_column=formula_column,
    )
    logging.info("Model saved at %s", vocab_path)


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on SMILES and molecular formulas")
    parser.add_argument("--dataset-name", default="jablonkagroup/pubchem-smiles-molecular-formula", 
                       help="HuggingFace dataset name")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--max-corpus-chars", type=int, default=100_000_000, 
                       help="Maximum corpus characters. Use -1 to process all data without limit.")
    parser.add_argument("--model-type", default="unigram", 
                       choices=["unigram", "bpe", "word", "char"],
                       help="SentencePiece model type (default: unigram)")
    parser.add_argument("--output-dir", default="data", help="Output directory for tokenizer")
    parser.add_argument("--vocab-model-name", default="sentencepiece_tokenizer.model", help="Name for the tokenizer model file")
    parser.add_argument("--sep-token", default="[SEP]", help="Separator token between molecular formula and SMILES")
    parser.add_argument("--smiles-column", default="smiles", help="Name of the SMILES column in the dataset")
    parser.add_argument("--formula-column", default="molecular_formula", help="Name of the molecular formula column in the dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, args.vocab_model_name)
    
    # Handle max_corpus_chars: convert -1 to None to indicate no limit
    max_corpus_chars = None if args.max_corpus_chars == -1 else args.max_corpus_chars
    
    train_tokenizer(
        args.dataset_name,
        vocab_path=vocab_path,
        vocab_size=args.vocab_size,
        max_corpus_chars=max_corpus_chars,
        model_type=args.model_type,
        sep_token=args.sep_token,
        smiles_column=args.smiles_column,
        formula_column=args.formula_column,
    )
    
    logging.info("Tokenizer training complete!")
    logging.info(f"Model saved to: {vocab_path}")
    if max_corpus_chars is None:
        logging.info("Used all available data (no character limit)")
    else:
        logging.info(f"Used up to {max_corpus_chars:,} characters from dataset")
    logging.info("To use the tokenizer:")
    logging.info("  import sentencepiece as spm")
    logging.info(f"  sp = smp.SentencePieceProcessor(model_file='{vocab_path}')")
    logging.info(f"Used columns: formula='{args.formula_column}', smiles='{args.smiles_column}'")


if __name__ == "__main__":
    main()
