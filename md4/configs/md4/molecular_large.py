from collections import abc

from ml_collections import config_dict

def get_config() -> config_dict.ConfigDict:
  """Default config."""

  config = config_dict.ConfigDict()

  # dataset configs
  config.vocab_size = 2048
  config.dataset = "pubchem_large"
  config.version = "1.0.7"
  config.training_shards = 256
  config.validation_shards = 8
  config.include_formula = True

  config.classes = -1
  config.max_length = 128
  config.tokenizer = "data/pubchem_large_tokenizer_2048"

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
  config.fp_bits = 4096
  config.fingerprint_dim = 4096
  config.fingerprint_mlp_layers = (2048, 512, 256, 64)  # Configurable SimpleMLP layers for fingerprint conditioning
  

  config.feature_dim = 64
  config.n_layers = 12
  config.ch_mult = (1,)  # not used
  config.n_dit_layers = 0  # not used
  config.dit_num_heads = 12  # not used
  config.dit_hidden_size = 768  # not used
  config.dropout_rate = 0.0
  config.multiple_of = 256

  config.num_heads = 12
  config.mlp_type = "glu"
  config.depth_scaled_init = True
  config.cond_type = "adaln_zero"

  config.learning_rate = 2e-4
  config.learning_rate_schedule = "cosine"
  config.warmup_steps = 2000
  config.weight_decay = 1e-06
  config.clip = 0.0
  config.b2 = 0.999
  config.num_epochs = -1
  config.ema_rate = 0.9999
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = 1_500_000
  # Evaluates for a full epoch if num_eval_steps==-1.
  config.num_eval_steps = 1000
  config.batch_size = 1024
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
  config.eval_every_steps = 20000
  config.checkpoint_every_steps = 20000
  config.checkpoint_keep_period = 200000

  # Single integer or tuple. If None will use (XManager ID, work unit).
  config.seed = 88

  # Number of workers for Grain loaders.
  config.grain_num_workers = 15
  config.grain_num_read_threads = 4
  config.grain_prefetch_buffer_size = 128

  config.trial = 0  # Dummy for repeated runs.
  config.test_in_colab = False
  return config