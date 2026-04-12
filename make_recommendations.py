import json
from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

import jax
import jax.numpy as jnp

FLAGS = flags.FLAGS
_PRODUCT_EMBED_ = flags.DEFINE_string("product_embed", 'product_embed.json', "Product embedding json.")
_SCENE_EMBED_ = flags.DEFINE_string("scene_embed", 'scene_embed.json', "Scene embedding json.")

def find_top_k(scene_embedding, product_embedding, k):
    
    pass

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

    print(product_embeddings.shape)

    jax.jit(find_top_k)


if __name__ == '__main__':
    app.run(main)
    pass