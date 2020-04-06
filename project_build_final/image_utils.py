# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:26:18 2020

@author: ealegre
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib as plt

# IMAGE UTILITIES - FUNCTION 1 - LOAD IMAGE

def load_img(img_path):
    # Read the image and convert the computation graph to an image format
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Setting the scale parameters to change the size of the image
    # Get the width and height of the image. Cast it to a float so it can be divided
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # Set the absolute maximum dimension for the image
    # max_dim = 1024
    max_dim = 512
    # Find which side is the longer side, this will be used to generate our scale
    max_side = max(shape)
    scale = max_dim / max_side
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape, method=tf.image.ResizeMethod.BILINEAR)

    img = img[tf.newaxis, :]
    return img

# IMAGE UTILITIES - FUNCTION 2 - DEPROCESS IMAGE

def deprocess_img(processed_img):
    processed_img = processed_img*255
    processed_img = np.array(processed_img, dtype=np.uint8)
    if np.ndim(processed_img)>3:
      assert processed_img.shape[0] == 1
      processed_img = processed_img[0]
    return Image.fromarray(processed_img)

# IMAGE UTILITIES - FUNCTION 3 - SAVE IMAGE

def save_img(best_img, path):
    img = Image.fromarray(best_img)
    img.save(path)
    
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)