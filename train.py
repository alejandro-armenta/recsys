from typing import Sequence, Tuple

from absl import flags
from absl import app
from absl import logging

import tensorflow as tf

import optax
import flax

from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state

import jax
import jax.numpy as knp


import numpy as np

FLAGS = flags.FLAGS

_INPUT_FILE=flags.DEFINE_string('input_file', 'fashion.json', 'Input cat json file.')

_IMAGE_DIRECTORY=flags.DEFINE_string('image_dir', 'artifacts/shop_the_look:v1', 'Directory containing downloaded images from the shop the look dataset.')

_NUM_NEG=flags.DEFINE_integer('num_neg', 5, 'How many negatives per positive.')

_LEARNING_RATE=flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')

_REGULARIZATION=flags.DEFINE_float('regularization', 0.1, 'Regularization.')

_OUTPUT_SIZE=flags.DEFINE_integer('output_size', 32, 'Size of output embedding.')

_BATCH_SIZE=flags.DEFINE_integer('batch_size', 16, 'Batch size.')

_LOG_EVERY_STEPS=flags.DEFINE_integer('log_every_steps', 100, 'Log every this step.')

_EVAL_EVERY_STEPS = flags.DEFINE_integer("eval_every_steps", 2000, "Eval every this step.")

_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 100000, "Checkpoint every this step.")

_MAX_STEPS = flags.DEFINE_integer("max_steps", 30000, "Max number of steps.")

_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "pinterest_stl_model", "Model name.")

_RESTORE_CHECKPOINT = flags.DEFINE_bool("restore_checkpoint", False, "If true, restore.")




tf.config.set_visible_devices([], 'GPU')



