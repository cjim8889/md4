from collections import abc

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Default config."""

  config = config_dict.ConfigDict()

  # dataset configs
  config.vocab_size = 1024
  config.dataset = "msg_finetune"
  config.classes = -1
  config.max_length = 128
  config.tokenizer = "data/pubchem_large_tokenizer"
  config.version = "1.0.6"

  config.min_frequency = 200
  config.pad_to_length = 128 # Not used
  config.atom_type_size = 0 # Not used

  config.task_type = "text"  # text or image
  config.model_type = "md4"
  config.data_shape = (config.max_length,)

  # timesteps: int or None
  config.timesteps = 1000
  # linear, cosine, poly[exponent], e.g., poly3
  config.noise_schedule = "cosine"
  config.outside_embed = True
  # t or none (removes time dependence)
  config.time_features = "t"
  config.cont_time = True
  config.fingerprint_adapter = True
  config.only_adapter = True
  config.raw_fingerprint_dim = 4096
  config.fingerprint_dim = 2048
  config.old_config = "md4/configs/md4/molecular.py"
  config.frozen = True
  config.partial_load = True
  
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


  config.feature_dim = 64
  config.n_layers = 12
  config.ch_mult = (1,)  # not used
  config.n_dit_layers = 0  # not used
  config.dit_num_heads = 12  # not used
  config.dit_hidden_size = 768  # not used
  config.dropout_rate = 0.02

  config.num_heads = 12
  config.mlp_type = "swiglu"
  config.depth_scaled_init = True
  config.cond_type = "adaln_zero"
  config.multiple_of = 256

  config.learning_rate = 1e-4
  config.learning_rate_schedule = "cosine"
  config.warmup_steps = 1000
  config.weight_decay = 0.0
  config.clip = 0.0
  config.b2 = 0.999
  config.num_epochs = -1
  config.ema_rate = 0.9999
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = 100_000
  # Evaluates for a full epoch if num_eval_steps==-1.
  config.num_eval_steps = 100
  config.batch_size = 512
  config.num_microbatches = 1
  config.per_device_batch_size = -1
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = False
  config.check_nans = False

  # Sampling
  # ancestral, mean, or topp
  config.sampler = "ancestral"
  # uniform, cosine
  config.sampling_grid = "uniform"
  # for topp sampler
  config.topp = 0.98

  config.log_loss_every_steps = 1000
  config.eval_every_steps = 2000
  config.checkpoint_every_steps = 2000
  config.checkpoint_keep_period = 6000

  # Single integer or tuple. If None will use (XManager ID, work unit).
  config.seed = 88

  # Number of workers for Grain loaders.
  config.grain_num_workers = 15
  config.grain_num_read_threads = 4
  config.grain_prefetch_buffer_size = 128

  config.trial = 0  # Dummy for repeated runs.
  config.test_in_colab = False
  return config