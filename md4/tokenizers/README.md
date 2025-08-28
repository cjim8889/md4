# Tokenizers Module

Clean, modular tokenizers for molecular SMILES processing. No bloat, just what works.

## Architecture

```
tokenizers/
├── base.py                  # Abstract base class defining tokenizer interface
├── sentencepiece_tokenizer.py  # TensorFlow SentencePiece implementation
├── smiles_tokenizer.py     # HuggingFace transformers wrapper
└── __init__.py             # Module exports and factory function
```

## Usage

### Direct Import

```python
from md4.tokenizers import SentencePieceTokenizer, SMILESTokenizer

# SentencePiece tokenizer (TensorFlow-based)
sp_tokenizer = SentencePieceTokenizer("path/to/model.model")
tokens = sp_tokenizer.encode("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")
decoded = sp_tokenizer.decode(tokens)

# HuggingFace tokenizer
hf_tokenizer = SMILESTokenizer("path/to/tokenizer", max_length=128)
tokens = hf_tokenizer.encode("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")
decoded = hf_tokenizer.decode(tokens)
```

### Factory Pattern

```python
from md4.tokenizers import create_tokenizer

# Create tokenizers using factory
tokenizer = create_tokenizer("sentencepiece", "path/to/model.model")
tokenizer = create_tokenizer("huggingface", "path/to/tokenizer", max_length=128)
```

## Interface

All tokenizers implement the `BaseTokenizer` interface:

- `vocab_size`: Vocabulary size
- `pad_id`: Padding token ID
- `unk_id`: Unknown token ID  
- `encode(text)`: Convert text to token IDs
- `decode(token_ids)`: Convert token IDs to text
- `batch_decode(token_ids)`: Decode multiple sequences

## Dependencies

- **SentencePieceTokenizer**: `tensorflow`, `tensorflow_text`
- **SMILESTokenizer**: `transformers`

## Design Principles

1. **Clean interfaces** - Minimal, consistent API across all tokenizers
2. **No magic** - Explicit behavior, no hidden state
3. **Type safety** - Clear input/output types
4. **Modularity** - Each tokenizer is independent
5. **Performance** - Efficient batch processing

## Testing

```bash
uv run python md4/tokenizers/test_tokenizers.py
```

## Migration from Legacy Code

Replace inline tokenizer classes with imports:

```python
# Old way
class SentencePieceTokenizer:
    def __init__(self, model_path):
        # ... 100 lines of code ...

# New way
from md4.tokenizers import SentencePieceTokenizer
```

That's it. Keep it simple, keep it working.