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

"""Deterministic input pipeline."""

from collections.abc import Sequence
import dataclasses
import os
from typing import Any, Union
import urllib.request
import zipfile

import grain.python as grain
import jax
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


# pylint: disable=g-import-not-at-top
try:
    import cv2
except ImportError:
    pass
try:
    from md4 import input_pipeline_pubchem, input_pipeline_pubchem_large, input_pipeline_msg_finetune, input_pipeline_pubchem_large_text
except ImportError as e:
    print("input_pipeline_pubchem not found", e)
FlatFeatures = dict[str, Any]

_DataSet = Union[grain.MapDataset, grain.DataLoader, grain.IterDataset]

_GPT2_TOKENIZER = "gpt2"
_OWT_DATASETS = dict(
    # OSS version. Please prepare the OWT datasets using the following command:
    # python ./prepare_openwebtext_data.py
    dataset_train_path=("./data_dir/openwebtext_splits_1024_train"),
    dataset_eval_path=("./data_dir/openwebtext_splits_1024_eval"),
)


class Text8Tokenizer:
    """Simple text8 tokenizer."""

    def __init__(self, num_extra_tokens=0):
        self.num_extra_tokens = num_extra_tokens

    @property
    def vocab_size(self):
        return 27 + self.num_extra_tokens

    @property
    def pad_token(self):
        return 26

    def encode(self, text):
        tokens = np.array([i - 97 for i in text], dtype=np.int32)
        tokens = np.where(tokens < 0, self.pad_token, tokens)
        return tokens

    def decode(self, tokens):
        tokens = np.where(np.equal(tokens, self.pad_token), 32 - 97, tokens) + 97
        text = tokens.astype(np.uint8).tobytes()
        return text.decode("utf-8")


def preprocess_text8(
    data_dir,
    doc_length: int = 512,
):
    """Load the 27-char text8 dataset."""
    if not os.path.exists(os.path.join(data_dir, "text8.train.txt")):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(os.path.join(data_dir, "text8.zip")):
            url = "http://mattmahoney.net/dc/text8.zip"
            print("Downloading text8 from URL {}.".format(url))
            urllib.request.urlretrieve(url, os.path.join(data_dir, "text8.zip"))
        with open(os.path.join(data_dir, "text8.zip"), "rb") as f:
            rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")
        splits = {
            "train": rawdata[:90000000],
            "valid": rawdata[90000000:95000000],
            "test": rawdata[95000000:],
        }
        for split, data in splits.items():
            with open(os.path.join(data_dir, "text8." + split + ".txt"), "w") as f:
                f.write(data)

    def load_text8_split(split: str):
        def _split_chars(arr):
            return tf.compat.v1.string_split(
                [arr], sep="", result_type="RaggedTensor"
            ).flat_values

        def _join_and_rename(x):
            text = tf.strings.reduce_join(x, axis=0)
            return {"text": text}

        path = os.path.join(data_dir, "text8." + split + ".txt")
        ds = tf.data.TextLineDataset(path).map(_split_chars).unbatch()
        ds = ds.batch(doc_length, drop_remainder=True)
        ds = ds.map(_join_and_rename)
        return ds

    # Define the builder.
    text8_builder = tfds.dataset_builders.store_as_tfds_dataset(
        name="text8",
        version="1.0.0",
        features=tfds.features.FeaturesDict(
            {
                "text": tfds.features.Text(),
            }
        ),
        split_datasets={
            "train": load_text8_split("train"),
            "valid": load_text8_split("valid"),
            "test": load_text8_split("test"),
        },
        config="text8",
        data_dir=data_dir,
        description="text8 dataset, document length 512.",
        file_format="array_record",
        disable_shuffling=True,
    )

    return text8_builder


class ChunkDataSource(grain.RandomAccessDataSource):
    """Chunk text data source."""

    def __init__(self, tensor, chunk_size=256, overlapping=False):
        self.chunk_size = chunk_size
        self.overlapping = overlapping
        tensor = tensor.encode("utf-8")
        if not overlapping:
            extra_len = len(tensor) % chunk_size
            if extra_len > 0:
                tensor = tensor[:-extra_len]
            self.tensor = np.array(list(tensor)).reshape(-1, chunk_size)
        else:
            self.tensor = tensor

    def __len__(self):
        if not self.overlapping:
            return self.tensor.shape[0]
        else:
            return len(self.tensor) - self.chunk_size + 1

    def __getitem__(self, record_key):
        if not self.overlapping:
            return {"text": self.tensor[record_key]}
        else:
            start_idx = record_key
            end_idx = record_key + self.chunk_size
            chunk = self.tensor[start_idx:end_idx]
            return {"text": chunk}

    def __repr__(self) -> str:
        return f"ChunkDataSource(len={len(self)},overlapping={self.overlapping})"


@dataclasses.dataclass
class Tokenize(grain.MapTransform):
    tokenizer: Text8Tokenizer

    def map(self, features):
        text = features["text"]
        features["text"] = self.tokenizer.encode(text)
        return features


@dataclasses.dataclass(frozen=True)
class DiscreteWithoutLabel(grain.MapTransform):
    """Discrete image data with zero labels."""

    def map(self, features):
        features["image"] = features["image"].astype(np.int32)
        if "label" in features:
            del features["label"]
        if "id" in features:
            del features["id"]
        return features


@dataclasses.dataclass(frozen=True)
class ResizeSmall(grain.MapTransform):
    """Resizes the smaller side to `size` keeping aspect ratio.

    Attr:
      size: Smaller side of an input image (might be adjusted if max_size given).
    """

    size: int

    def map(self, features: FlatFeatures) -> FlatFeatures:
        image = features["image"]
        size = self.size
        image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_AREA)
        features["image"] = image.astype(np.int32)
        return features


@dataclasses.dataclass(frozen=True)
class CentralSquareCrop(grain.MapTransform):
    """Makes a square central square crop of a given size."""

    def map(self, features: FlatFeatures) -> FlatFeatures:
        image = features["image"]
        h, w = image.shape[:2]
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        image = image[top : top + size, left : left + size, :]
        features["image"] = image
        return features


def get_data_shape(config):
    return config.data_shape


@dataclasses.dataclass(frozen=True)
class DropFeatures(grain.MapTransform):
    feature_names: Sequence[str]

    def map(self, features: FlatFeatures) -> FlatFeatures:
        for feature_name in self.feature_names:
            del features[feature_name]
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


def get_num_train_steps(config: config_dict.ConfigDict) -> int:
    """Calculates the total number of training steps."""
    if config.num_train_steps > 0:
        return config.num_train_steps
    # From the beginning. We first shard the data (shard by process_count), then
    # combine all epochs, batch for all local devices.
    # In all steps we would drop the remainder (hence the use of integer
    # division).
    # When start_index is 0 the train_ds.cardinality() and num_train_steps should
    # be equivalent.
    if config.task_type == "image":
        tfds_info = tfds.builder(config.dataset).info
        num_train_records = tfds_info.splits["train"].num_examples
        return int(num_train_records // jax.process_count() * config.num_epochs) // (
            config.per_device_batch_size * jax.local_device_count()
        )
    else:
        raise NotImplementedError()


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[_DataSet, dict[str, _DataSet], dict[str, Any]]:
    """Create Grain data loaders for training and evaluation.

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

    if config.dataset == "text8":
        seq_len = config.data_shape[0]
        # Current train/valid format only support length of 256
        assert seq_len == 256

        with tf.io.gfile.GFile(os.path.join("./data", "text8", "text8.zip"), "rb") as f:
            rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")
        splits = {
            "train": rawdata[:90000000],
            "valid": rawdata[90000000:95000000],
            "test": rawdata[95000000:],
        }

        tokenizer = Text8Tokenizer(num_extra_tokens=0)
        train_transformations = [Tokenize(tokenizer)]
        train_source = ChunkDataSource(
            splits["train"], chunk_size=seq_len, overlapping=True
        )

        eval_transformations = [Tokenize(tokenizer)]
        eval_source = {
            k: ChunkDataSource(splits[k], chunk_size=seq_len) for k in ["valid", "test"]
        }
        info["tokenizer"] = tokenizer
        info["rawdata"] = rawdata
    elif config.dataset == "openwebtext":
        # we need to pretrain a GPT2 size model with context length of 1024.
        seq_len = config.data_shape[0]
        assert seq_len == 1024
        train_transformations = [ParseFeatures(data_column="text")]
        eval_transformations = [ParseFeatures(data_column="text")]

        train_table_path = _OWT_DATASETS["dataset_train_path"]
        train_source = grain.ArrayRecordDataSource(paths=train_table_path)

        eval_source = {
            "owt_eval": grain.ArrayRecordDataSource(
                paths=_OWT_DATASETS["dataset_eval_path"]
            ),
        }

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)
        info["tokenizer"] = tokenizer
    elif config.dataset == "pubchem":
        # Use the dedicated PubChem input pipeline
        (
            train_source,
            train_transformations,
            eval_source,
            eval_transformations,
            pubchem_info,
        ) = input_pipeline_pubchem.create_pubchem_datasets(config, seed)
        info.update(pubchem_info)
    elif config.dataset == "pubchem_large":
        train_dataset, eval_dataset, pubchem_info = input_pipeline_pubchem_large.create_pubchem_datasets(config, seed)
        info.update(pubchem_info)
        return train_dataset, eval_dataset, info
    elif config.dataset == "pubchem_large_text":
        train_dataset, eval_dataset, pubchem_info = input_pipeline_pubchem_large_text.create_pubchem_datasets(config, seed)
        info.update(pubchem_info)
        return train_dataset, eval_dataset, info
    elif config.dataset == "msg_finetune":
        train_dataset, eval_dataset, msg_info = input_pipeline_msg_finetune.create_msg_finetune_datasets(config, seed)
        info.update(msg_info)
        return train_dataset, eval_dataset, info
    elif (
        config.dataset.startswith("mnist")
        or config.dataset.startswith("cifar")
        or config.dataset.startswith("downsampled_imagenet")
    ):
        data_source = tfds.data_source(config.dataset)
        train_transformations = [DiscreteWithoutLabel()]
        train_source = data_source["train"]
        eval_transformations = [DiscreteWithoutLabel()]
        eval_source = {k: v for k, v in data_source.items() if k != "train"}
    elif config.dataset == "class_cond_imagenet64":
        data_source = tfds.data_source("imagenet2012")
        train_transformations = [
            CentralSquareCrop(),
            ResizeSmall(64),
            DropFeatures(("file_name",)),
        ]
        train_source = data_source["train"]
        eval_transformations = [
            CentralSquareCrop(),
            ResizeSmall(64),
            DropFeatures(("file_name",)),
        ]
        eval_source = {"validation": data_source["validation"]}
    else:
        raise NotImplementedError("Unsupported datasets.")

    train_loader = grain.load(
        source=train_source,
        shuffle=True,
        seed=seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        transformations=train_transformations,
        batch_size=process_batch_size,
        worker_count=config.grain_num_workers,
        read_options=grain.ReadOptions(
            num_threads=config.grain_num_read_threads,
            prefetch_buffer_size=config.grain_prefetch_buffer_size,
        ),
    )

    if config.eval_pad_last_batch:
        raise NotImplementedError(
            "BatchWithPadElements is not implemented in PyGrain yet."
        )
    else:
        drop_remainder = True
        shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)

    eval_loaders = {}
    for split in eval_source:
        eval_loader = grain.load(
            source=eval_source[split],
            num_epochs=1,
            shard_options=shard_options,
            transformations=eval_transformations,
            batch_size=process_eval_batch_size,
            worker_count=0,
            drop_remainder=drop_remainder,
            read_options=grain.ReadOptions(
                num_threads=config.grain_num_read_threads,
                prefetch_buffer_size=config.grain_prefetch_buffer_size,
            ),
        )
        eval_loaders[split] = eval_loader

    return train_loader, eval_loaders, info
