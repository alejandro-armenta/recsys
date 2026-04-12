
import os

import json

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import flax

import numpy as np


import pin_util

import models

import ip

from tqdm import tqdm

FLAGS = flags.FLAGS

_INPUT_FILE=flags.DEFINE_string('input_file', 'fashion.json', 'Input cat json file.')

_IMAGE_DIRECTORY=flags.DEFINE_string('image_dir', 'artifacts/shop_the_look:v1', 'Directory containing downloaded images from the shop the look dataset.')

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "artifacts/pinterest_stl_model_rc1:v0/pinterest_stl.model",
    "Model name.")

_OUTPUT_SIZE = flags.DEFINE_integer("output_size", 64, "Size of embeddings.") 

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8, "Batch size.")

_OUTDIR = flags.DEFINE_string("out_dir", "/tmp", "Output directory.")

def main(argv):
    del argv
    
    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()

    #primero va la escena y luego el producto!
    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)

    logging.info("Found %d valid scene product pairs." % len(scene_product))

    unique_scenes=set(x[0] for x in scene_product)
    unique_products=set(x[1] for x in scene_product)

    unique_scenes = np.array(list(unique_scenes))
    unique_products = np.array(list(unique_products))

    model = models.STLModel(output_size=_OUTPUT_SIZE.value)

    state = None

    with open(_MODEL_NAME.value, 'rb') as f:
        print(f)

        data = f.read()

        state = flax.serialization.from_bytes(model, data)

    assert(state != None)

    #print(unique_scenes)

    def get_scene_embed(x):
        return model.apply(state['params'], x, method=models.STLModel.get_scene_embed)
    
    def get_product_embed(x):
        return model.apply(state['params'], x, method=models.STLModel.get_product_embed)
        pass

    ds = tf.data.Dataset.from_tensor_slices(unique_scenes).map(ip.process_image_with_id)

    ds = ds.batch(_BATCH_SIZE.value, drop_remainder=True)


    it = ds.as_numpy_iterator()

    print(len(ds))

    scene_dict = {}
    
    other = tqdm(it, total=len(ds))


    for id, image in other:
        #8 ids y 8 imagenes
        result = get_scene_embed(image)

        for i in range(_BATCH_SIZE.value):
            
            current_id = id[i].decode('utf-8')
            
            tmp = np.array(result[i])

            current_result = [float(tmp[j]) for j in range(tmp.shape[0])]

            scene_dict.update({current_id:current_result})

    with open('scene_embed.json', 'w') as scene_file:
        json.dump(scene_dict, scene_file)
    
    
    ds = tf.data.Dataset.from_tensor_slices(unique_products).map(ip.process_image_with_id)

    ds = ds.batch(_BATCH_SIZE.value, drop_remainder=True)

    it = ds.as_numpy_iterator()

    print(len(ds))

    product_dict = {}
    
    other = tqdm(it, total=len(ds))

    for id, image in other:
        #8 ids y 8 imagenes
        result = get_product_embed(image)

        for i in range(_BATCH_SIZE.value):
            
            current_id = id[i].decode('utf-8')
            
            tmp = np.array(result[i])

            current_result = [float(tmp[j]) for j in range(tmp.shape[0])]

            product_dict.update({current_id:current_result})

    with open('product_embed.json', 'w') as scene_file:
        json.dump(product_dict, scene_file)
    






    

            

    






if __name__ == '__main__':
    app.run(main)