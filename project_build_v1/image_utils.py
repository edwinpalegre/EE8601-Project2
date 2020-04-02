# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:26:13 2020

@author: ealegre
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing import image as tf_image


def img_load(img_path):
    max_dim = 512
    img = Image.open(img_path)
    img_size = max(img.size)
    scale = max_dim/img_size
    # Possibly try Image.BILINEAR instead?
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.BILINEAR)
    img = tf_image.img_to_array(img)
    
    # Image needs to be broadcasted so that it has a dimension for the batch
    img = np.expand_dims(img, axis=0)
    return img
    
def load_process_img(img_path):
    img = img_load(img_path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
        
    # Perform the inverse of preprocessing stage
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_results(best_img, path):
    img = Image.fromarray(best_img)
    img.save(path)