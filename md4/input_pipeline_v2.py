# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input pipeline for grain based datasets."""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any

from absl import logging
import datasets
import datasets.distributed
from etils import epath
import grain.python as grain
import jax
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


_GPT2_TOKENIZER = "gpt2"
_OWT_DATASETS = dict(
    # OSS version. Please prepare the OWT datasets using the following command:
    # python ./prepare_openwebtext_data.py
    dataset_train_path=("./data_dir/openwebtext_splits_1024_train"),
    dataset_eval_path=("./data_dir/openwebtext_splits_1024_eval"),
)


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support."""

  def __init__(
      self,
      dataset: datasets.IterableDataset,
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      generate_padding_example: bool,
      max_target_length: int,
      data_column_name: str,
  ):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.generate_padding_example = generate_padding_example
    self.max_target_lenth = max_target_length
    self.data_column_name = data_column_name
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [
        dataloading_host_index * self.num_threads + i
        for i in range(self.num_threads)
    ]
    self.datasets = [
        datasets.distributed.split_dataset_by_node(
            dataset, world_size=self.n_shards, rank=x
        )
        for x in self.dataset_shards
    ]
    self.data_iters = []
    self.out_of_data = False

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      print(
          f"WARNING: Inefficient dataloading. Your train or eval dataset"
          f" contains {self.n_shards} shards, smaller than number of host"
          " loading data. This is known to lead to inefficient dataloading."
          " see"
          " https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    new_shard = (
        self.dataset_shards[idx]
        + self.dataloading_host_count * self.num_threads
    )
    if new_shard < self.n_shards:
      print(
          f"Updating host {self.dataloading_host_index} dataset {idx}, was on"
          f" shard {self.dataset_shards[idx]}"
      )
      print(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = datasets.distributed.split_dataset_by_node(
          self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx]
      )
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      print(
          f"Run out of shards on host {self.dataloading_host_index}, shard"
          f" {self.dataset_shards[idx]} is not available"
      )
      self.out_of_data = True
      if self.generate_padding_example:
        print(
            f"Host {self.dataloading_host_index} will start generating all-0"
            " padding examples until step number is met."
        )

  def __len__(self):
    """Return length of the HF dataset.

    Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned
    """
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset doesn't support random access by index.

    The next item in the iterator is returned.

    Args:
      index: The index of the item to return.

    Returns:
      The next item in the iterator.
    """
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(threading.current_thread().name.split("_")[1])

    while True:
      try:
        if self.out_of_data:
          if self.generate_padding_example:
            return {
                self.data_column_name: np.zeros(
                    self.max_target_lenth, dtype=np.int32
                )
            }
          else:
            return None
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, features):
    return {"text": features[self.column_name].decode()}


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input."""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, features):
    return {
        "text": np.asarray(features[self.column_name], dtype=np.int32),
    }


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def map(self, data):
    return {
        "text": data[0]["text"],
        "text_segmentation": data[1]["text"],
        "text_position": data[2]["text"],
    }


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length."""

  def __init__(self, max_length):
    self.max_length = max_length

  def map(self, data):
    """map to each element."""

    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)

    data["text_segmentation"] = np.ones(data["text"].shape, dtype=np.int32)
    data["text_position"] = np.arange(data["text"].shape[0], dtype=np.int32)
    for key, _ in data.items():
      data[key] = _pad(data[key], self.max_length)
    return data


@dataclasses.dataclass
class TokenizeAndTrim(grain.MapTransform):
  """Tokenize and trim features to sequence length."""

  # pylint: disable=attribute-defined-outside-init
  feature_names: Sequence[str]
  sequence_length: Sequence[int]
  tokenizer: Any
  add_bos: bool
  add_eos: bool

  def map(self, features: dict[str, Any]) -> dict[str, Any]:
    """Maps to each element."""
    for feature_name, sequence_length in zip(
        self.feature_names, self.sequence_length, strict=True
    ):
      text = features[feature_name]
      token_ids = self.tokenizer(text)["input_ids"]
      if self.add_bos:
        token_ids = [self.tokenizer.bos_token_id] + token_ids

      if self.add_eos:
        token_ids = token_ids[: sequence_length - 1]
        token_ids = token_ids + [self.tokenizer.eos_token_id]
      else:
        token_ids = token_ids[:sequence_length]

      features[feature_name] = np.asarray(token_ids, dtype=np.int32)
    return features


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example."""

  def __init__(self, data_column):
    self.data_column = data_column

  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(
          example,
          {
              self.data_column: tf.io.FixedLenSequenceFeature(
                  [], dtype=tf.int64, allow_missing=True
              )
          },
      )
      return parsed

    return _parse(features)




def tokenization(example, hf_tokenizer, max_length, column_name):
  """Tokenize a HuggingFace dataset."""
  return hf_tokenizer(
      example[column_name], truncation=True, max_length=max_length
  )


def load_fineweb_edu_hf_source():
  """Loads fineweb_edu data source from HuggingFace.

  Returns:
    A grain data source for fineweb_edu.
  """
  tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
  fw = datasets.load_dataset(
      "HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True
  )
  fw = fw.map(
      tokenization,
      batched=True,
      fn_kwargs={
          "hf_tokenizer": tokenizer,
          "max_length": 1023,
          "column_name": "text",
      },
  )
  fw = fw.select_columns(["input_ids"]).rename_column("input_ids", "text")
  return HFDataSource(fw, 0, 1, 1, False, 1024, "text")


def compile_transformations(
    seq_len,
    tokenizer,
    data_column="text",
    add_bos=False,
    add_eos=True,
    packing=True,
    drop_remainder=True,
    process_batch_size=32,
):
  """Collects transformations for the grain input pipeline."""
  # Normalize: convert bytes to string ready for tokenization
  operations = []
  operations.append(NormalizeFeatures(data_column))
  operations.append(
      TokenizeAndTrim([data_column], [seq_len], tokenizer, add_bos, add_eos)
  )

  # Pack and Batch examples.
  if packing:
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=process_batch_size,
            length_struct={data_column: seq_len},
        )
    )
    operations.append(ReformatPacking())
  else:
    operations.append(PadToMaxLength(seq_len))
    operations.append(
        grain.Batch(
            batch_size=process_batch_size,
            drop_remainder=drop_remainder,
        )
    )
  return operations


def compile_hf_transformations(
    seq_len,
    data_column="text",
    process_batch_size=32,
):
  """Collects transformations for the grain input pipeline."""
  operations = []
  operations.append(HFNormalizeFeatures(data_column))
  operations.append(
      grain.experimental.PackAndBatchOperation(
          batch_size=process_batch_size,
          length_struct={data_column: seq_len},
      )
  )
  operations.append(ReformatPacking())
  return operations


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[grain.DataLoader, dict[str, grain.DataLoader], dict[str, Any]]:
  """Create Grain data loaders for training and evaluation.

  For the same seed and config this will return the same datasets.
  The user is responsible to save()/load() the dataset iterators (for training)
  or calling reset() to restart the iterator (for eval).

  Args:
    config: Configuration to use.
    seed: Seed for shuffle and random operations in the training dataset.

  Returns:
    A tuple with the training dataset loader, the evaluation dataset
    loader, and a dictionary of other infos.
  """
  info = {}
  assert config.batch_size % jax.process_count() == 0
  process_batch_size = config.batch_size // jax.process_count()
  eval_batch_size = config.get("eval_batch_size", config.batch_size)
  process_eval_batch_size = eval_batch_size // jax.process_count()

  if  config.dataset == "fineweb_edu":
    # we need to pretrain a GPT2 size model with context length of 1024.
    seq_len = config.data_shape[0]
    assert seq_len == 1024

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)

    load_fineweb_edu_source = load_fineweb_edu_hf_source
    train_source = load_fineweb_edu_source()
    transformations = compile_hf_transformations(
        seq_len,
        data_column="text",
        process_batch_size=process_batch_size,
    )
    eval_sources = {
        "owt_eval": grain.ArrayRecordDataSource(
            paths=_OWT_DATASETS["dataset_eval_path"]
        ),
        # "fwe_eval": eval_source,
    }
    eval_transformations = {
        "owt_eval": [
            ParseFeatures(data_column="text"),
            grain.Batch(
                batch_size=process_eval_batch_size,
                drop_remainder=True,
            ),
        ],
        # "fwe_eval": transformations,
    }
    info["tokenizer"] = tokenizer

  else:
    raise NotImplementedError("Unsupported datasets.")
  index_sampler = grain.IndexSampler(
      num_records=len(train_source),
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      shuffle=True,
      seed=seed,
  )
  train_loader = grain.DataLoader(
      data_source=train_source,
      operations=transformations,
      sampler=index_sampler,
      worker_count=config.grain_num_workers,
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1024),
  )

  if config.eval_pad_last_batch:
    raise NotImplementedError(
        "BatchWithPadElements is not implemented in PyGrain yet."
    )
  else:
    drop_remainder = True
    shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)

  eval_loaders = {}
  for split in eval_sources:
    eval_loader = grain.load(
        source=eval_sources[split],
        num_epochs=1,
        shard_options=shard_options,
        transformations=eval_transformations[split],
        # For now, we do not parallelize the evaluation, because there is a
        # bug on DataLoader.__iter__ when used with Jax.
        worker_count=0,
        read_options=grain.ReadOptions(prefetch_buffer_size=1024),
    )
    eval_loaders[split] = eval_loader

  return train_loader, eval_loaders, info
