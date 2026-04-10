from typing import Sequence

from flax import linen as nn


class CNN(nn.Module):
    filters : Sequence[int]
    @nn.compact
    def __call__(self, x, train:bool = True):

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
        
        return x


class STLModel(nn.Module):

    outpu_size : int

    def setup(self):
        default_filter = [16, 32, 64, 128]
        self.scene_cnn = CNN(filters=default_filter)
    
    def __call__(self, scene, pos_product, neg_product, train:bool = True):
        pass

