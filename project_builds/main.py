# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 01:36:16 2020

@author: ealegre
"""

import tensorflow as tf

import image_utils as Image
import model as Model

import os
import glob

files = glob.glob('output/*')
for f in files:
    os.remove(f)

content_path = 'input/content/edwin.jpg'
style_path = 'input/style/ironman.jpg'

content = Image.img_load(content_path).astype('uint8')
style = Image.img_load(style_path).astype('uint8')

# Content layer where will pull our feature maps
content_layers = ['block4_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

best, best_loss = Model.run_style_transfer(content_path, style_path, num_iterations=1000)

Image.save_results(best, 'output/output.jpg')