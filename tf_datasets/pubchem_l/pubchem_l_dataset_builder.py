"""pubchem_l dataset."""

import numpy as np
import polars as pl
import tensorflow_datasets as tfds

from .rdkit_utils import process_smiles


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pubchem_l dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(pubchem_l): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'smiles': tfds.features.Text(),
            'safe': tfds.features.Text(),
            'atom_types': tfds.features.Tensor(shape=(None,), dtype=np.int8),
            'fingerprints': tfds.features.Tensor(shape=(2048,), dtype=np.bool_),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('safe', 'fingerprints', 'atom_types'),  # Set to `None` to disable
        homepage='https://github.com/cjim8889/md4',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    df = pl.read_parquet('hf://datasets/jablonkagroup/pubchem-smiles-molecular-formula/data/train-*.parquet')
    total_size = len(df['smiles'])
    train_size = int(total_size * 0.95)
    val_size = int(total_size * 0.05)
    train_df = df.select(range(train_size))
    val_df = df.select(range(train_size, train_size + val_size))
    # TODO(pubchem_l): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train_df),
        'val': self._generate_examples(val_df),
    }

  def _generate_examples(self, df):
    """Yields examples."""
    for smile in df['smiles']:
        output = process_smiles(smile)
        if output is not None:
            yield smile, output
