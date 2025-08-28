from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    """Default config."""

    config = config_dict.ConfigDict()

    # dataset configs
    config.vocab_size = 2048
    config.dataset = "msg_finetune"
    config.version = "1.1.0"
    config.training_shards = 1
    config.validation_shards = 1
    config.include_formula = True

    config.classes = -1
    config.max_length = 128
    config.tokenizer = "data/pubchem_large_tokenizer_2048"

    config.min_frequency = 200
    config.pad_to_length = 128  # Not used
    config.atom_type_size = 0  # Not used

    config.task_type = "text"  # text or image
    config.model_type = "md4"
    config.data_shape = (config.max_length,)

    # timesteps: int or None
    config.timesteps = 1000
    # linear, cosine, poly[exponent], e.g., poly3
    config.noise_schedule = "linear"
    config.outside_embed = True
    # t or none (removes time dependence)
    config.time_features = "t"
    config.cont_time = True
    config.fp_bits = 4096
    config.fingerprint_dim = 4096
    config.fingerprint_mlp_layers = (
        2048,
        1024,
        512,
        256,
    )  # Configurable SimpleMLP layers for fingerprint conditioning

    ### Finetuning
    config.old_checkpoint_steps = 2820000
    config.fingerprint_adapter = True
    config.only_adapter = True
    config.raw_fingerprint_dim = 4096
    config.old_config = "md4/configs/md4/molecular_xtra_large.py"
    config.frozen = False
    config.partial_load = False

    # Frozen parameter configuration
    # Note: paths are matched using 'in' operator against the parameter path tuple
    # e.g., for parameter path ('transformer', 'layer_0', 'fp_adapter', 'kernel')
    # - "fp_adapter" in path would match (exact string match in tuple)
    # - "layer_0" in path would match
    # - "transformer" in path would match
    config.frozen_paths = []  # Paths to freeze (empty means freeze all except unfrozen_paths)
    config.unfrozen_paths = ["fp_adapter"]  # Paths to keep unfrozen

    # Adapter initialization paths (for special initialization)
    config.adapter_init_paths = ["fp_adapter"]

    ###

    config.feature_dim = 256
    config.n_layers = 10
    config.ch_mult = (1,)  # not used
    config.n_dit_layers = 0  # not used
    config.dit_num_heads = 12  # not used
    config.dit_hidden_size = 768  # not used
    config.dropout_rate = 0.0
    config.multiple_of = 256

    config.num_heads = 8
    config.n_kv_heads = 4
    config.mlp_type = "glu"
    config.depth_scaled_init = True
    config.cond_type = "adaln_zero"

    config.learning_rate = 3e-4
    config.learning_rate_schedule = "cosine"
    config.warmup_steps = 2000
    config.weight_decay = 0.0
    config.clip = 0.0
    config.b2 = 0.999
    config.num_epochs = -1
    config.ema_rate = 0.9999
    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs.
    config.num_train_steps = 25000
    # Evaluates for a full epoch if num_eval_steps==-1.
    config.num_eval_steps = 1000
    config.batch_size = 512
    config.num_microbatches = 2
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

    config.log_loss_every_steps = 1000
    config.eval_every_steps = 1000
    config.checkpoint_every_steps = 1000
    config.checkpoint_keep_period = 20000

    # Single integer or tuple. If None will use (XManager ID, work unit).
    config.seed = 88

    #

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
