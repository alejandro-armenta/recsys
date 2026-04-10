from typing import Sequence, Tuple

from absl import flags
from absl import app
from absl import logging

import tensorflow as tf

#import optax
#import flax

#from flax import linen as nn
#from flax.training import checkpoints
#from flax.training import train_state

import jax
import jax.numpy as knp

import numpy as np

import pin_util
import ip
import models

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

def generate_triplets(scene_product: Sequence[Tuple[str,str]], num_neg: int)->Sequence[Tuple[str,str,str]]:

    count = len(scene_product)
    key = jax.random.PRNGKey(0)

    test = []
    train = []

    for i in range(count):
        scene, pos = scene_product[i]

        is_test = i % 10 == 0

        key,subkey=jax.random.split(key)

        neg_indices =jax.random.randint(subkey, [num_neg], 0, count - 1)

        for neg_idx in neg_indices:
            _,neg = scene_product[neg_idx]
            
            if is_test:
                test.append((scene,pos,neg))
            else:
                train.append((scene,pos,neg))

    return train, test 

def shuffle_array(key, x):
    num = len(x)
    #son indices a x
    to_swap = jax.random.randint(key, [num], 0, num - 1)

    return [x[i] for i in to_swap]
    
          
def main(argv):
    del argv

    config = {
        'learning_rate': _LEARNING_RATE.value,
        'regularization': _REGULARIZATION.value,
        'output_size': _OUTPUT_SIZE.value,
    }

    print(tf.config.list_physical_devices('GPU'))
    print(jax.devices())

    
    #tf.config.set_visible_devices([], 'GPU')

    tf.compat.v1.enable_eager_execution()

    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)

    train,test=generate_triplets(scene_product, _NUM_NEG.value)

    logging.info('Train triplets: %d', len(train))
    logging.info('Test triplets: %d', len(test))

    key = jax.random.PRNGKey(0)

    train = shuffle_array(key, train)
    test = shuffle_array(key, test)

    #print(type(train))

    train = np.array(train)
    test = np.array(test)

    train_ds = ip.create_dataset(train).repeat() 
    train_ds = train_ds.batch(_BATCH_SIZE.value).prefetch(tf.data.AUTOTUNE)

    test_ds = ip.create_dataset(test).repeat() 
    test_ds = test_ds.batch(_BATCH_SIZE.value).prefetch(tf.data.AUTOTUNE)

    stl=models.STLModel(outpu_size=config['output_size'])

    key,subkey=jax.random.split(key)

    train_it = train_ds.as_numpy_iterator()

    #son las 3 fotos
    x = next(train_it)
    #entonces primero se llama setup y despues se llama call y se pasan los parametros x[0], x[1], x[2]

    stl.init(subkey, x[0], x[1], x[2])



    #print(ds.shape)


if __name__ == "__main__":
    app.run(main)



