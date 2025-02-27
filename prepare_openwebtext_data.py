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

"""Prepares the input pipeline for OpenWebText (OWT).

This script tokenizes the OWT dataset and splits it into train and eval sets.
The train and eval sets are saved as ArrayRecord files.
"""

from array_record.python import array_record_module
import datasets
import numpy as np
import tensorflow as tf
import tqdm
import transformers


source = datasets.load_dataset(
    "Skylion007/openwebtext", name="plain_text", split="train", streaming=True
)

_GPT2_TOKENIZER = "gpt2"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)

ArrayRecordWriter = array_record_module.ArrayRecordWriter
ArrayRecordReader = array_record_module.ArrayRecordReader


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


ds_output_file_train = "./data_dir/openwebtext_splits_1024_train"
ds_output_file_eval = "./data_dir/openwebtext_splits_1024_eval"

n_examples = 8013769  # tiny: 2; small: 10_000; full: 8013769
save_every_examples = 10_000
block_size = 1024  # size of the chunk

data_iter = (iter(source))

all_tokens = []
count = 0
count_per_save = 0
eval_chunks = []

writer_train = ArrayRecordWriter(ds_output_file_train, "group_size:1")
writer_eval = ArrayRecordWriter(ds_output_file_eval, "group_size:1")

for example in data_iter:
  tokens = tokenizer(example["text"])["input_ids"]
  all_tokens.extend(tokens + [tokenizer.eos_token_id])
  count += 1
  count_per_save += 1

  # pause to save when having tokenized enough examples for saving.
  if count_per_save >= save_every_examples:
    # save to disk
    saved_length = (len(all_tokens) // block_size) * block_size
    chunks = [
        all_tokens[i : i + block_size]
        for i in range(0, saved_length, block_size)
    ]

    print(f"\nsaving chunks @ {count}th example mark...")
    np.random.shuffle(chunks)
    num_eval = int(len(chunks) * 0.02)  # put 2% of chunks into eval split.
    for eval_i in tqdm.tqdm(range(num_eval)):
      feature = {
          "text": _int64_feature(chunks[eval_i]),
      }
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
      )
      writer_eval.write(example_proto.SerializeToString())

    for train_i in tqdm.tqdm(range(num_eval, len(chunks))):
      feature = {
          "text": _int64_feature(chunks[train_i]),
      }
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
      )
      writer_train.write(example_proto.SerializeToString())

    # prepare for the next round of tokenize-n-save.
    all_tokens = all_tokens[saved_length:]
    count_per_save = 0

  # stop when having tokenized enough examples for total #.
  if count >= n_examples:
    # save to disk
    saved_length = (len(all_tokens) // block_size) * block_size
    chunks = [
        all_tokens[i : i + block_size]
        for i in range(0, saved_length, block_size)
    ]

    print(f"\nsaving chunks @ {count}th example mark...")
    np.random.shuffle(chunks)
    num_eval = int(len(chunks) * 0.02)  # put 2% of chunks into eval split.
    for eval_i in tqdm.tqdm(range(num_eval)):
      feature = {
          "text": _int64_feature(chunks[eval_i]),
      }
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
      )
      writer_eval.write(example_proto.SerializeToString())

    for train_i in tqdm.tqdm(range(num_eval, len(chunks))):
      feature = {
          "text": _int64_feature(chunks[train_i]),
      }
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
      )
      writer_train.write(example_proto.SerializeToString())
    break

writer_train.close()
writer_eval.close()
