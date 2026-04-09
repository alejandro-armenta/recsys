
from typing import Sequence, Tuple

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

def create_dataset(triplet: Sequence[Tuple[str,str,str]]):

    tf.data.Dataset.from_tensor_slices(triplet)
    pass


create_dataset(('ale','jorge','maria'))