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
    input_files: Union[str, list, None] = None,
):
    """Train SentencePiece tokenizer from HuggingFace dataset or list of input files.
    Args:
      dataset: HuggingFace dataset (ignored if input_files is provided)
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
      input_files: Union[str, list, None]: If provided, use these files directly instead of dataset.
        Can be a single file path (str) or list of file paths. Files should contain text data.
    Returns:
      path to the trained sentencepiece vocabulary model.
    """
    if model_path.startswith("gs://"):
        abs_model_path = model_path
    else:
        abs_model_path = os.path.abspath(os.path.expanduser(model_path))
    
    # Handle input files: either use provided files or dump dataset to file
    if input_files is not None:
        # Use provided input files directly
        if isinstance(input_files, str):
            # Single file path
            input_file_list = [input_files]
        else:
            # List of file paths
            input_file_list = input_files
        
        # Validate that all files exist
        for file_path in input_file_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Join file paths with comma for SentencePiece input
        input_files_str = ",".join(input_file_list)
        logging.info(f"Using input files: {input_files_str}")
        fname = None  # No temporary file to clean up
        
        # Calculate total characters for logging (optional)
        actual_chars = 0
        try:
            for file_path in input_file_list:
                with open(file_path, 'r', encoding='utf-8') as f:
                    actual_chars += len(f.read())
            logging.info(f"Total characters in input files: {actual_chars}")
        except Exception as e:
            logging.warning(f"Could not calculate total characters: {e}")
            
    else:
        # Use dataset and dump to temporary file (original behavior)
        fname, actual_chars = _dump_hf_data_to_textfile(
            dataset, maxchars=maxchars, sep_token=sep_token,
            smiles_column=smiles_column, formula_column=formula_column
        )
        input_files_str = fname
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
            f"--input={input_files_str}",
            f"--vocab_size={vocab_size}",
            f"--character_coverage={character_coverage}",
            f"--model_prefix={model_fp.name}",
            f"--model_type={model_type}",
            f"--user_defined_symbols={','.join(user_defined_symbols)}",
            "--input_sentence_size=100000000",
            "--add_dummy_prefix=false",
            "--num_threads=128",
            "--unk_piece=[UNK]",
            "--bos_piece=[BEGIN]",
            "--eos_piece=[END]",
            "--pad_piece=[PAD]",
            "--train_extremely_large_corpus=true",
            "--shuffle_input_sentence=true",
            "--max_sentencepiece_length=32",
            "--split_by_unicode_script=false",
            "--split_by_number=false",
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
        # Only clean up fname if it was created (not using input_files)
        if fname is not None:
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
    input_files: Union[str, list, None] = None,
):
    """tokenizer training function
    Args:
        dataset_name: str: HuggingFace dataset name (ignored if input_files is provided)
        vocab_path: str: path to save the trained tokenizer model
        vocab_size: int: vocabulary size
        max_corpus_chars: int: maximum number of characters to use from corpus. If None or -1, use all data.
        model_type: str: type of sentencepiece model ('unigram', 'bpe', 'word', 'char').
        sep_token: str: separator token between molecular formula and SMILES
        smiles_column: str: name of SMILES column in dataset
        formula_column: str: name of molecular formula column in dataset
        input_files: Union[str, list, None]: If provided, use these files directly instead of dataset.
            Can be a single file path (str) or list of file paths. Files should contain text data.
    """
    if input_files is not None:
        # Use provided input files directly
        logging.info(f"Using provided input files: {input_files}")
        ds = None  # No dataset needed
    else:
        # Load dataset from HuggingFace
        logging.info("Loading dataset from HuggingFace...")
        ds = datasets.load_dataset(dataset_name, split="train").with_format("polars")
        logging.info("Dataset loaded successfully")
    
    logging.info("Training SentencePiece tokenizer...")
    vocab_path = _train_sentencepiece(
        ds,
        vocab_size=vocab_size,
        maxchars=max_corpus_chars,
        model_path=vocab_path,
        model_type=model_type,
        sep_token=sep_token,
        smiles_column=smiles_column,
        formula_column=formula_column,
        input_files=input_files,
    )
    logging.info("Model saved at %s", vocab_path)


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on SMILES and molecular formulas")
    parser.add_argument("--dataset-name", default="jablonkagroup/pubchem-smiles-molecular-formula", 
                       help="HuggingFace dataset name (ignored if --input-files is provided)")
    parser.add_argument("--input-files", nargs='+', 
                       help="List of input text files to use instead of HuggingFace dataset. "
                            "Files should contain text data with formula[SEP]smiles format.")
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
        input_files=args.input_files,
    )
    
    logging.info("Tokenizer training complete!")
    logging.info(f"Model saved to: {vocab_path}")
    if args.input_files:
        logging.info(f"Used input files: {args.input_files}")
    else:
        if max_corpus_chars is None:
            logging.info("Used all available data (no character limit)")
        else:
            logging.info(f"Used up to {max_corpus_chars:,} characters from dataset")
    logging.info("To use the tokenizer:")
    logging.info("  import sentencepiece as spm")
    logging.info(f"  sp = smp.SentencePieceProcessor(model_file='{vocab_path}')")
    if not args.input_files:
        logging.info(f"Used columns: formula='{args.formula_column}', smiles='{args.smiles_column}'")


if __name__ == "__main__":
    main()
