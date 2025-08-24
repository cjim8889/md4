import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    """Default config."""

    config = config_dict.ConfigDict()

    # wandb configs
    config.enable_wandb = True
    config.wandb_project = "md4"

    # mixed precision configs
    config.dtype = jnp.bfloat16
    config.param_dtype = jnp.float32

    # profiler
    config.start_profiler = False
    
    # Multi-host configuration
    config.initialize_multihost = True  # Enable multi-host JAX initialization

    # dataset configs
    config.vocab_size = 3000
    config.dataset = "pubchem_large_text"
    config.version = "1.1.0"
    config.training_shards = 128
    config.validation_shards = 4
    config.include_formula = True
    
    # Data directory configuration
    config.parquet_data_dir = "data/pubchem_large/data"  # Directory containing train-*.parquet files
    config.tfrecord_data_dir = "/mnt/data/pubchem_large_text"  # Directory to read/write TFRecord files
    
    # High-entropy data loading configuration for maximum diversity between epochs
    config.cycle_length = 16  # Number of TFRecord files to interleave concurrently
    config.block_length = 4   # Number of consecutive elements to take from each file
    config.file_shuffle_buffer = 1000      # File-level shuffle buffer size
    config.record_shuffle_buffer = 10000   # Record-level shuffle buffer size (training)
    config.batch_shuffle_buffer = 50       # Batch-level shuffle buffer size

    config.classes = -1
    config.max_length = 128
    config.tokenizer = "data/sentencepiece_tokenizer_bpe_3000_newcorpus.model"

    config.min_frequency = 200

    config.task_type = "text"  # text or image
    config.model_type = "md4"
    config.data_shape = (config.max_length,)

    # timesteps: int or None
    config.timesteps = 500
    # linear, cosine, poly[exponent], e.g., poly3
    config.noise_schedule = "linear"
    config.outside_embed = True
    # t or none (removes time dependence)
    config.time_features = "t"
    config.cont_time = True
    config.fp_bits = 4096
    config.fingerprint_dim = 4096

    config.feature_dim = 256
    config.n_layers = 20
    config.dropout_rate = 0.0
    config.multiple_of = 32

    config.num_heads = 8
    config.n_kv_heads = 4
    config.mlp_type = "glu"
    config.depth_scaled_init = True
    config.cond_type = "adaln_zero"
    
    # Cross-attention configuration
    config.use_cross_attention = True  # Set to True to enable cross-attention
    config.cross_attention_layers = 8  # None for all layers, or number for first N layers
    config.cross_attention_proj_dim = 256
    config.cross_conditioning_seq_length = 16  # Sequence length for cross-conditioning reshape

    config.learning_rate = 3e-4
    config.learning_rate_schedule = "cosine"
    config.warmup_steps = 2000
    config.weight_decay = 1e-4
    config.clip = 0.0
    config.b2 = 0.999
    config.num_epochs = -1
    config.ema_rate = 0.
    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs.
    config.num_train_steps = 500_000
    # Evaluates for a full epoch if num_eval_steps==-1.
    config.num_eval_steps = 1000
    config.batch_size = 3456
    config.num_microbatches = 3
    config.per_device_batch_size = -1
    # If batches should be added to evaluate the entire dataset.
    config.eval_pad_last_batch = False
    config.check_nans = False

    # Sampling
    # ancestral, mean, or topp
    config.sampler = "topp"
    # uniform, cosine
    config.sampling_grid = "cosine"
    # for topp sampler
    config.topp = 0.98

    config.log_loss_every_steps = 500
    config.eval_every_steps = 10000
    config.checkpoint_every_steps = 10000
    config.checkpoint_keep_period = 200000
    
    # Checkpoint directory configuration
    config.checkpoint_dir = "gs://metal-repeater-411410-tpu-checkpoints/1B_3000_vocab_linear_glu_bf16_expt/checkpoints"

    # Single integer or tuple. If None will use (XManager ID, work unit).
    config.seed = 88

    config.trial = 0  # Dummy for repeated runs.
    
    # Device mesh configuration for distributed training
    config.mesh_config = config_dict.ConfigDict()
    config.mesh_config.mesh_shape = (16, )  # Mesh shape, e.g., (2, 4) for 8 devices
    config.mesh_config.mesh_axis_names = ("data", )  # Names for mesh axes
    
    # Global array multiplier for make_global_array function
    # This should be the size of the model axis (first dimension in mesh_shape)
    config.global_array_multiplier = config.mesh_config.mesh_shape[0]  # model axis size = 2

    # Logical sharding configuration
    # Maps logical axis names (used in nn.with_logical_partitioning) to physical mesh axes
    # 'data': shard across data parallel dimension
    # 'model': shard across model parallel dimension  
    # None/unspecified: leave unsharded
    config.logical_axis_rules = [
        ('batch', 'data'),           # Batch dimension -> data parallel
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
        # Sequence/time dimensions are left unsharded for efficiency
    ]
    
    return config
