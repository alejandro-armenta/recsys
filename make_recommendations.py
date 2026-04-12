import os

import json

from typing import Any, Tuple, Dict

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

import jax
import jax.numpy as jnp

import pin_util

FLAGS = flags.FLAGS
_PRODUCT_EMBED_ = flags.DEFINE_string("product_embed", 'product_embed.json', "Product embedding json.")
_SCENE_EMBED_ = flags.DEFINE_string("scene_embed", 'scene_embed.json', "Scene embedding json.")
_TOP_K = flags.DEFINE_integer('top_k', 10, 'Number of top scoring products to return per scene')
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "./tmp", "Location to write output.")
_MAX_RESULTS = flags.DEFINE_integer("max_results", 100, "Max scenes to score.")


def find_top_k(scene_embedding, product_embeddings, k):

    #1 64 9224 64
    scores = scene_embedding * product_embeddings

    scores = jnp.sum(scores, axis=-1)

    scores_and_indices = jax.lax.top_k(scores, k)

    return scores_and_indices

def local_file_to_pin_url(filename):
  """Converts a local filename to a pinterest url."""
  key = filename.split("/")[-1]
  key = key.split(".")[0]
  url = pin_util.key_to_url(key)
  result = "<img src=\"%s\">" % url
  return result


def save_results(
        filename:str, 
        scene_key:str, 
        scores_and_indices:Tuple[Any, Any], 
        index_to_key:Dict[int, str]):
    
    scores, indices = scores_and_indices


    scores, indices = np.array(scores), np.array(indices)
    
    with open(filename, 'w') as f:
        f.write('<HTML>\n')
        scene_image = local_file_to_pin_url(scene_key)
        f.write(f"Nearest neighbors to {scene_image}<br>\n")

        for i in range(scores.shape[0]):
            idx = indices[i]
            product_key = index_to_key[idx]
            product_img = local_file_to_pin_url(product_key)
            f.write(f"Rank {i+1} Score {scores[i]:f}<br>{product_img}<br>\n")
        
        f.write("</HTML>\n")
        


     

def main(argv):
    del argv

    tf.compat.v1.enable_eager_execution()

    with open(_PRODUCT_EMBED_.value, 'r') as f:
        product_dict = json.load(f)

    with open(_SCENE_EMBED_.value, 'r') as f:
        scene_dict = json.load(f)
        
    index_to_key = {}
    
    product_embeddings = []

    for i, (key, vec) in enumerate(product_dict.items()):
        #print(i, key, vec)
        
        index_to_key.update({i:key})

        product_embeddings.append(np.array(vec))

    product_embeddings = jnp.stack(product_embeddings, axis=0)

    #print(product_embeddings.shape)

    top_k_finder = jax.jit(find_top_k, static_argnames=['k'])

    for i, (key, vec) in enumerate(scene_dict.items()):
        other = jnp.array(vec)

        scene_embed = jnp.expand_dims(other, axis=0)

        scores_and_indices = top_k_finder(scene_embed, product_embeddings, _TOP_K.value)

        filename = os.path.join(_OUTPUT_DIR.value, f"{i:05d}.html")

        save_results(filename, key, scores_and_indices, index_to_key)

        if i > _MAX_RESULTS.value:
            break


if __name__ == '__main__':
    app.run(main)
    pass