
from typing import Sequence, Tuple

import tensorflow as tf

def normalize_image(img):
    img = tf.cast(img, dtype=tf.float32)
    img = (img / 255.0) - 0.5
    return img

def process_image(x):
    x = tf.io.read_file(x)
    x = tf.io.decode_jpeg(x, channels=3)
    x = tf.image.resize_with_crop_or_pad(x, 512,512)
    x = normalize_image(x)
    return x

def process_image_with_id(id):
    image = process_image(id)
    return id, image

def process_triplet(x):
    return (process_image(x[0]), process_image(x[1]), process_image(x[2]))

def create_dataset(triplet: Sequence[Tuple[str,str,str]]):

    ds = tf.data.Dataset.from_tensor_slices(triplet)
    ds = ds.map(process_triplet)

    return ds
    