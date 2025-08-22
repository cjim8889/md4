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

"""Main file for running the example.

This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.
"""

import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import jax
import tensorflow.compat.v2 as tf
from absl import app, flags, logging

# Required import to setup work units when running through XManager.
from clu import platform
from ml_collections import config_flags

from md4 import sharded_train_v2, train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("olddir", None, "Old checkpoint directory for partial loading.")
flags.DEFINE_boolean("sharded", False, "Whether to use sharded training.")
flags.mark_flags_as_required(["config", "workdir"])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
    del argv

    tf.enable_v2_behavior()
    # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    if FLAGS.sharded:
        sharded_train_v2.train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.olddir)
    else:
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.olddir)


if __name__ == "__main__":
    # Provide access to --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    run_main = app.run
    run_main(main)
