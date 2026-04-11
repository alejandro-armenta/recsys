
#import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
#os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
import jax.numpy as jnp

import numpy as np

import wandb

import pin_util
import ip
import models

#from tqdm import tqdm

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

_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 10, "Checkpoint every this step.")

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

def train_step(state, scene, pos_product, neg_product, regularization, batch_size):

    def loss_fn(params):
        #aqui tiene que estar mi modelo!
        result, new_model_state = state.apply_fn(params, scene, pos_product, neg_product, True, mutable=['batch_stats'])

        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))

        def reg_fn(embedding):
            return nn.relu(
                jnp.sqrt(jnp.sum(jnp.square(embedding), axis=-1)) - 1
                )
            
        reg_loss = reg_fn(result[2]) + reg_fn(result[3]) + reg_fn(result[4])

        reg_loss = jnp.sum(reg_loss)

        return (triplet_loss + regularization * reg_loss) / batch_size
        
    grad_fn = jax.value_and_grad(loss_fn)

    #esta madre evalua y calcula grad!
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)

    return new_state, loss

def eval_step(state, scene, pos_product, neg_product):
    
    def loss_fn(params):
        #aqui tiene que estar mi modelo!
        result, new_model_state = state.apply_fn(
            params, 
            scene, 
            pos_product, 
            neg_product, 
            True, 
            mutable=['batch_stats']
            )

        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))

        return triplet_loss
    
    loss = loss_fn(state.params)

    return loss
    
          
def main(argv):
    del argv

    config = {
        'learning_rate': _LEARNING_RATE.value,
        'regularization': _REGULARIZATION.value,
        'output_size': _OUTPUT_SIZE.value,
    }

    run = wandb.init(config=config, project='recsys-pinterest')

    print(tf.config.list_physical_devices('GPU'))
    print(jax.devices())

    
    #tf.config.set_visible_devices([], 'GPU')

    tf.compat.v1.enable_eager_execution()

    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)

    train,test=generate_triplets(scene_product, _NUM_NEG.value)

    num_test = len(test)
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

    stl=models.STLModel(output_size=wandb.config.output_size)

    key,subkey=jax.random.split(key)

    train_it = train_ds.as_numpy_iterator()
    test_it = test_ds.as_numpy_iterator()

    #son las 3 fotos
    x = next(train_it)
    #entonces primero se llama setup y despues se llama call y se pasan los parametros x[0], x[1], x[2]

    #esto te pasa los parametros
    params = stl.init(subkey, x[0], x[1], x[2])

    print(params.keys())

    tx = optax.adam(learning_rate=wandb.config.learning_rate)
    
    state = train_state.TrainState.create(apply_fn=stl.apply, params=params, tx=tx)

    train_step_fn = jax.jit(train_step)

    eval_step_fn = jax.jit(eval_step)

    init_step = state.step
    regularization = wandb.config.regularization

    batch_size = _BATCH_SIZE.value

    eval_steps = int(num_test / batch_size)

    losses = []
    #0,30000
    for i in range(init_step, _MAX_STEPS.value + 1):
        #este toma el que sigue
        batch = next(train_it)
        scene = batch[0]
        pos_product = batch[1]
        neg_product = batch[2]

        metrics = {
            'step' : state.step
        }

        state, loss = train_step_fn(state, scene, pos_product, neg_product, regularization, batch_size)

        losses.append(loss)

        if i % _EVAL_EVERY_STEPS.value == 0 and i > 0:
            eval_loss = []
            for j in range(eval_steps):
                ebatch = next(test_it)
                escene = ebatch[0]
                epos_product = ebatch[1]
                eneg_product = ebatch[2]

                loss = eval_step_fn(state, escene, epos_product, eneg_product)

                eval_loss.append(loss)
            
            eval_loss = jnp.mean(jnp.array(eval_loss)) / batch_size
            metrics.update({'eval_loss':eval_loss})
                
            

        if i % _LOG_EVERY_STEPS.value == 0 and i > 0:
            mean_loss = jnp.mean(jnp.array(losses))
            losses = []
            metrics.update({'train_loss':mean_loss})
            wandb.log(metrics)
            logging.info(metrics)

    logging.info("Saving as %s", _MODEL_NAME.value)

    data = flax.serialization.to_bytes(state)

    metadata = { "output_size" : wandb.config.output_size}
    
    artifact = wandb.Artifact(
        name=_MODEL_NAME.value,
        metadata=metadata,
        type="model")
    
    with artifact.new_file("pinterest_stl.model", "wb") as f:
        f.write(data)

    run.log_artifact(artifact)
        



if __name__ == "__main__":
    app.run(main)



