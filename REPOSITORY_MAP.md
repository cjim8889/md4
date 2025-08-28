# Repository File Map

## Repository Overview
This is a **masked discrete diffusion model** implementation for molecular generation and text processing using JAX/Flax. The system implements a transformer-based architecture for generating molecules represented as SMILES strings using a discrete diffusion process.

## Core Architecture Files

### Entry Points
- **`md4/main.py`** - Primary execution entry point with CLI argument parsing
  - Handles configuration loading and dispatches to training modes
  - Sets up JAX backend and TensorFlow GPU hiding
  - Routes to either standard or sharded training based on `--sharded` flag

### Training Infrastructure
- **`md4/train.py`** - Standard training loop with pmap parallelization
  - Implements loss computation, training/eval steps, and sampling
  - Handles EMA updates, metric collection, and checkpointing
  - Supports fingerprint conditioning and SMILES validity calculation
  - Contains full training and evaluation orchestration

- **`md4/sharded_train.py`** - FSDP (Fully Sharded Data Parallel) training implementation
  - Advanced sharded training with device mesh configuration
  - Microbatch processing with shard_map optimization
  - Multi-host JAX support with preemption handling
  - Memory-efficient gradient accumulation across shards

### Core Model Components

#### Diffusion Models
- **`md4/models/diffusion/md4.py`** - Main MD4 diffusion model implementation
  - **MaskingSchedule**: Implements noise schedules (cosine, linear, polynomial)
  - **MD4**: Core masked discrete diffusion model class
    - Forward/prior sampling for adding noise
    - Conditional sampling for sequence initialization
    - Multiple sampling strategies (ancestral, top-p, mean)
    - Fingerprint adapter integration
    - Cross-attention conditioning support

- **`md4/models/diffusion/genmd4.py`** - Generative MD4 variant

#### Transformer Architecture
- **`md4/networks/transformer.py`** - LLAMA2-style transformer implementation
  - **Attention**: Multi-head self-attention with RoPE positional encoding
  - **CrossAttention**: Cross-attention for conditioning on external features
  - **FeedForward**: SwiGLU/GeGLU activation functions
  - **TransformerBlock**: Complete transformer layer with optional cross-attention
  - **Transformer**: Full model with AdaLN conditioning support

- **`md4/networks/sharded_transformer.py`** - Sharded transformer for distributed training
- **`md4/networks/adapters.py`** - Adapter layers for fingerprint conditioning

#### Backward Process
- **`md4/models/backward.py`** - Discrete classifier for the backward diffusion process
- **`md4/models/utils.py`** - Model utility functions and factory methods

### Data Pipeline

#### Input Processing
- **`md4/input_pipeline.py`** - Main data pipeline dispatcher
  - Routes to specific dataset implementations
  - Calculates training steps and data shapes

- **`md4/input_pipeline_pubchem_large.py`** - PubChem large dataset processing
- **`md4/input_pipeline_pubchem_large_text.py`** - Text-based PubChem processing  
- **`md4/input_pipeline_msg_finetune.py`** - Message-based fine-tuning dataset

#### Tokenization
- **`md4/tokenizers/`** - Clean, modular tokenizer implementations
  - **`base.py`** - Abstract tokenizer interface
  - **`sentencepiece_tokenizer.py`** - TensorFlow SentencePiece implementation
  - **`smiles_tokenizer.py`** - HuggingFace transformers wrapper
  - **`test_tokenizers.py`** - Tokenizer test suite
  - **`README.md`** - Comprehensive tokenizer documentation

### Sampling and Generation
- **`md4/sampling.py`** - Sample generation functions
  - **generate()**: pmap-based sampling with conditioning
  - **simple_generate()**: Single-device sampling
  - **reconstruct()**: Latent reconstruction from timestep t

- **`md4/binary_search.py`** - Efficient binary search algorithms
  - Float32 and int32 binary search implementations  
  - Top-k and top-p masking for probability distributions
  - Optimized for TPU/GPU execution with minimal communication

## Configuration System

### Model Configurations
- **`md4/configs/md4/`** - MD4 model configuration variants
  - **`molecular.py`** - Base molecular generation config (1000 timesteps, cosine schedule)
  - **`molecular_*.py`** - Various model size and training configurations
    - Size variants: `molecular_large.py`, `molecular_xtra_large.py`
    - Training modes: `molecular_finetune.py`, `molecular_xtra_large_finetune.py`
    - Sharding configs: `molecular_xtra_large_sp*.py` (sp = sharded parallel)
    - Precision variants: `bf16`, `fp32` mixed precision options
    - Multi-host configs: `multi_host.py` variants

## Utility Modules

### Training Utilities
- **`md4/utils/state_utils.py`** - Training state management
  - TrainState dataclass definition
  - Conditioning extraction from batches
  - Sharded and standard train state creation
  - Parameter freezing and adapter initialization
  - Metrics collection setup

- **`md4/utils/checkpoint_utils.py`** - Checkpoint management
  - Standard and partial checkpoint loading
  - Checkpoint manager configuration
  - Multi-directory checkpoint handling

- **`md4/utils/learning_rate.py`** - Learning rate scheduling
- **`md4/utils/wandb_writer.py`** - Weights & Biases integration

### Chemical Utilities  
- **`md4/utils/rdkit_utils.py`** - RDKit molecular utilities
  - SMILES validation and property calculation
  - Chemical descriptor computation

- **`md4/utils/pubchem_worker.py`** - PubChem data processing workers

### General Utilities
- **`md4/utils/utils.py`** - General utility functions
  - Batch reshaping and image grid generation
  - RNG management and loss conversion utilities

- **`md4/utils/safe_utils.py`** - Safe computation utilities

## Key Features

### Diffusion Process
- **Masked Discrete Diffusion**: Uses vocabulary-size mask token for corrupting sequences
- **Multiple Noise Schedules**: Cosine, linear, and polynomial schedules
- **Continuous/Discrete Time**: Configurable time parameterization
- **Antithetic Sampling**: Improved training efficiency

### Conditioning Systems
- **Fingerprint Conditioning**: Molecular fingerprint integration with adapter layers
- **Cross-Attention**: External feature conditioning through cross-attention layers
- **AdaLN Conditioning**: Adaptive Layer Normalization for class/property conditioning

### Sampling Strategies
- **Ancestral Sampling**: Standard generative sampling
- **Top-p Sampling**: Nucleus sampling with binary search optimization
- **Mean Sampling**: Two-step sampling approach

### Training Features
- **FSDP Support**: Fully sharded data parallel training for large models
- **Microbatching**: Memory-efficient gradient accumulation
- **EMA Parameters**: Exponential moving average for stable generation
- **Mixed Precision**: BFloat16/Float32 training support
- **Multi-host Training**: Distributed training across multiple machines

### Molecular Applications
- **SMILES Generation**: Chemical structure generation as SMILES strings
- **Fingerprint Integration**: Chemical fingerprint conditioning
- **Validity Metrics**: Automated SMILES validation using RDKit
- **Property Conditioning**: Generate molecules with specific properties

## Development Patterns

### Sharding Strategy
The codebase implements a comprehensive sharding strategy:
- **Data Parallel**: Batch dimension sharded across "data" axis
- **Model Parallel**: Model weights sharded across "model" axis  
- **Logical Partitioning**: Automatic sharding specification using logical axis names
- **Mesh Configuration**: Flexible device mesh setup for multi-dimensional parallelism

### Memory Optimization
- **Gradient Accumulation**: Microbatch processing to fit large models
- **Parameter Donation**: Memory reuse through donated arguments
- **Mixed Precision**: BFloat16 computation with Float32 accumulation
- **Checkpointing**: Efficient state persistence with preemption handling

### Extensibility
- **Modular Design**: Clear separation between models, data, and training
- **Configuration-Driven**: ML Collections for flexible hyperparameter management
- **Adapter Patterns**: Easy integration of new conditioning modalities
- **Factory Functions**: Consistent object creation patterns