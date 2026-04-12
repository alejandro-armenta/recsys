from typing import Sequence

from flax import linen as nn
import jax.numpy as jnp

class CNN(nn.Module):
    filters : Sequence[int]
    output_size : int

    @nn.compact
    def __call__(self, x, train:bool = True):
        
        #print(x.shape)

        for filter in self.filters:

            #downsamples 2x
            residual = nn.Conv(filter, (3,3), (2,2))(x)
            x = nn.Conv(filter, (3,3), (2,2))(x)

            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = nn.swish(x)

            x = nn.Conv(filter, (1,1), (1,1))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = nn.swish(x)

            x = nn.Conv(filter, (1,1), (1,1))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)

            x = residual + x

            #downsamples 2x
            x = nn.avg_pool(x, (3,3), strides=(2,2), padding='SAME')

            #print(x.shape)
        
        
        x = jnp.mean(x, axis=(1,2))

        #print(x.shape)

        x = nn.Dense(self.output_size, dtype=jnp.float32)(x)
        
        #print(x.shape)
        return x


class STLModel(nn.Module):

    output_size : int

    def setup(self):
        default_filter = [16, 32, 64, 128]
        
        #two cnn towers!
        self.scene_cnn = CNN(filters=default_filter, output_size=self.output_size)

        self.product_cnn = CNN(filters=default_filter, output_size=self.output_size)
    
    def get_scene_embed(self, scene):
        return self.scene_cnn(scene, False)

    def get_product_embed(self, product):
        return self.product_cnn(product, False)

    def __call__(self, scene, pos_product, neg_product, train:bool = True):
        #cada imagen se vuelve un embedding
        scene_embed = self.scene_cnn(scene, train)

        pos_product_embed = self.product_cnn(pos_product, train)

        pos_score = scene_embed * pos_product_embed

        pos_score = jnp.sum(pos_score, axis=-1)

        neg_product_embed = self.product_cnn(neg_product, train)

        neg_score = scene_embed * neg_product_embed

        neg_score = jnp.sum(neg_score, axis=-1)

        return pos_score, neg_score, scene_embed, pos_product_embed, neg_product_embed


        

        

