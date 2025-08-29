import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    """Default config."""

    config = config_dict.ConfigDict()

    # wandb configs - for experiment tracking and monitoring
    config.enable_wandb = True  # Enable Weights & Biases logging for metrics/loss tracking
    config.wandb_project = "md4"  # W&B project name for organizing runs

    # mixed precision configs - for memory efficiency and training speed
    config.dtype = jnp.bfloat16  # Computation dtype - bfloat16 reduces memory usage by ~50%
    config.param_dtype = jnp.float32  # Parameter storage dtype - keeps full precision for stability

    # profiler - for performance debugging and optimization
    config.start_profiler = False  # Whether to start JAX profiler for performance analysis
    
    # Multi-host configuration - for distributed training across multiple machines
    config.initialize_multihost = True  # Enable multi-host JAX initialization for TPU pods

    # dataset configs - defines the training data and vocabulary
    config.vocab_size = 4000  # Vocabulary size for tokenizer - determines embedding table size
    config.dataset = "pubchem_large_text"  # Dataset type - controls which input pipeline is used
    config.version = "1.2.0"  # Dataset version for reproducibility and data tracking
    config.training_shards = 160  # Number of training data shards for parallel processing
    config.validation_shards = 6  # Number of validation data shards
    config.include_formula = True  # Whether to include molecular formula in training sequences
    
    # Data directory configuration - paths for data storage and processing
    config.parquet_data_dir = "data/pubchem_large/data"  # Source directory with raw parquet files
    config.tfrecord_data_dir = "/mnt/data/pubchem_large_text"  # Directory for processed TFRecord files
    config.interior_frac = 0.5  # Fraction of padding tokens placed in sequence interior vs edges
    config.num_variants = 2  # Number of augmented variants per training example

    # High-entropy data loading configuration - ensures maximum diversity between epochs
    config.cycle_length = 16  # Number of TFRecord files to interleave concurrently for data mixing
    config.block_length = 4   # Number of consecutive elements to take from each file before switching
    config.file_shuffle_buffer = 1000      # Buffer size for shuffling file order
    config.record_shuffle_buffer = 1000000   # Buffer size for shuffling records within files (training)
    config.batch_shuffle_buffer = 50       # Final shuffle buffer for batched examples

    config.classes = -1  # Number of classification classes (-1 for autoregressive text modeling)
    config.max_length = 160  # Maximum sequence length for input tokens
    config.tokenizer = "data/sentencepiece_tokenizer_safe_bpe_1500_split.model"  # Path to tokenizer model

    config.min_frequency = 200  # Minimum token frequency threshold for vocabulary filtering

    config.task_type = "text"  # Task type - determines which model architecture to use
    config.model_type = "md4"  # Model architecture type - selects MD4 transformer variant
    config.data_shape = (config.max_length,)  # Shape of input data tensors
    
    # SMILES validity calculation - for evaluating molecular generation quality
    config.calculate_smiles_validity = True  # Whether to compute SMILES validity during sampling

    # Diffusion model parameters - controls the denoising process
    config.timesteps = 250  # Number of diffusion timesteps for noise schedule
    config.noise_schedule = "linear"  # Type of noise schedule: linear, cosine, poly[n]
    config.outside_embed = True  # Whether to embed time features outside main model
    config.time_features = "t"  # Type of time encoding features (t=timestep, none=no time)
    config.cont_time = True  # Whether to use continuous time formulation
    config.fp_bits = 4096  # Number of bits in molecular fingerprints for conditioning
    config.fingerprint_dim = 4096  # Dimension of fingerprint embeddings

    # Core transformer architecture parameters
    config.feature_dim = 256  # Hidden dimension of transformer layers - controls model width
    config.n_layers = 10  # Number of transformer layers - controls model depth
    config.dropout_rate = 0.0  # Dropout probability for regularization (0.0 = no dropout)
    config.multiple_of = 256  # Constraint to make layer dimensions multiples of this for efficiency

    # Multi-head attention configuration
    config.num_heads = 8  # Number of attention heads in multi-head attention
    config.n_kv_heads = 4  # Number of key-value heads for grouped query attention (memory efficient)
    config.mlp_type = "glu"  # MLP activation type: glu, swiglu, geglu for feed-forward layers
    config.depth_scaled_init = True  # Whether to use depth-scaled parameter initialization
    config.cond_type = "adaln_zero"  # Conditioning mechanism type for diffusion model
    config.norm_type = "rmsnorm"  # Normalization type: auto, layernorm, rmsnorm
    
    # Cross-attention configuration - for conditioning on molecular fingerprints
    config.use_cross_attention = True  # Enable cross-attention between sequence and conditioning
    config.cross_attention_layers = None  # None=all layers, int=first N layers use cross-attention
    config.cross_attention_proj_dim = 256  # Projection dimension for cross-attention mechanism
    config.cross_conditioning_seq_length = 32  # Sequence length for reshaping cross-conditioning data

    # Training hyperparameters - controls optimization and learning dynamics
    config.learning_rate = 3e-4  # Base learning rate for optimizer
    config.learning_rate_schedule = "cosine"  # LR schedule type: cosine, linear, constant
    config.warmup_steps = 2000  # Number of warmup steps for learning rate scheduling
    config.weight_decay = 1e-2  # L2 regularization weight for AdamW optimizer
    config.scale_by_muon = True  # Whether to use Muon optimizer scaling (experimental)
    config.clip = 0.0  # Gradient clipping norm (0.0 = no clipping)
    config.b2 = 0.999  # Beta2 parameter for Adam optimizer (momentum decay)
    config.num_epochs = -1  # Number of training epochs (-1 = use num_train_steps instead)
    config.ema_rate = 0.  # Exponential moving average decay rate (0.0 = disabled)
    
    # Training schedule and batch configuration
    config.num_train_steps = 200_000  # Total number of training steps (used when num_epochs=-1)
    config.num_eval_steps = 1000  # Number of evaluation steps per eval round (-1 = full epoch)
    config.batch_size = 4096  # Global batch size across all devices
    config.num_microbatches = 2  # Number of gradient accumulation steps for memory efficiency
    config.per_device_batch_size = -1  # Auto-calculated from global batch size and device count
    config.eval_pad_last_batch = False  # Whether to pad final evaluation batch to full size
    config.check_nans = False  # Whether to check for NaN values during training

    # Sampling configuration - for generation and evaluation
    config.sampler = "topp"  # Sampling method: ancestral, mean, or topp (nucleus sampling)
    config.sampling_grid = "cosine"  # Time grid for sampling trajectory: uniform, cosine
    config.topp = 0.98  # Top-p threshold for nucleus sampling (only used with topp sampler)

    # Logging and checkpointing schedule
    config.log_loss_every_steps = 500  # Frequency of logging training metrics to console/wandb
    config.eval_every_steps = 5000  # Frequency of running model evaluation
    config.checkpoint_every_steps = 5000  # Frequency of saving model checkpoints
    config.checkpoint_keep_period = 200000  # Retention period for old checkpoints (in steps)
    
    # Checkpoint directory configuration - can be local path or GCS bucket
    # config.checkpoint_dir = "gs://metal-repeater-411410-tpu-checkpoints/1B_3000_vocab_linear_glu_bf16_expt/checkpoints"

    # Reproducibility and experiment tracking
    config.seed = 88  # Random seed for reproducible training (int or tuple)
    config.trial = 0  # Trial number for repeated runs with same config
    
    # Device mesh configuration - defines distributed training topology
    config.mesh_config = config_dict.ConfigDict()
    config.mesh_config.mesh_shape = (16, )  # Mesh shape: (data_parallel,) for 16 devices
    config.mesh_config.mesh_axis_names = ("data", )  # Names for mesh axes (data parallel only)
    
    # Global array multiplier - for distributed array initialization
    config.global_array_multiplier = config.mesh_config.mesh_shape[0]  # Size of first mesh dimension
    
    # Logical sharding configuration - maps model dimensions to physical device axes
    # Controls how tensors are partitioned across devices for memory and computation efficiency
    config.logical_axis_rules = [
        ('batch', 'data'),           # Batch dimension sharded across data-parallel devices
        # Model parallel sharding rules (currently disabled for data-parallel only setup):
        # ('hidden', 'model'),         # Hidden/embedding dimensions -> model parallel
        # ('attn_qkv', 'model'),      # Attention Q/K/V projections -> model parallel  
        # ('attn_o', 'model'),        # Attention output projection -> model parallel
        # ('ff_mlp', 'model'),        # Feed-forward MLP layers -> model parallel
        # ('embed_vocab', 'model'),   # Vocabulary embeddings -> model parallel
        # ('input_embed', 'model'),   # Input embeddings -> model parallel
        # ('cross_attn', 'model'),    # Cross-attention projections -> model parallel
        # ('cond', 'model'),          # Conditioning layers -> model parallel
        # ('cond_input', 'model'),    # Conditioning input layers -> model parallel
        # ('cond_hidden', 'model'),   # Conditioning hidden layers -> model parallel
        # ('cond_output', 'model'),   # Conditioning output layers -> model parallel
        # ('vocab', 'model'),         # Vocabulary output layers -> model parallel
        # Note: Sequence/time dimensions left unsharded for computational efficiency
    ]
    
    return config
