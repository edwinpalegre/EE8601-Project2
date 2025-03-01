# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:13:23 2020

@author: ealegre
"""

import tensorflow as tf
import numpy as np
import image_utils as Image
from tensorflow.python.keras import models

# Define which layers are to be used for this model. These layers are defined in Section 
# 3 of Gatys' paper
content_layers = ['block4_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():
    """
    Creates the model with intermediate layer access
    
    This function will load in teh VGG19 model used in Gatys' paper, with access to the 
    intermediate layers. A new model will be generated by using these layers that will take an 
    input image and return an output from the intermediate layers from the VGG19 model

    Returns
    -------
    A Keras model taht takes inputs and outputs of the style and content intermediate layers

    """
    
    # Load the VGG19 model that was trained using ImageNet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling='avg')
    # Since this model is already pretrained, we will set it to not trainable
    vgg.trainable = False
    # Retrieve outputs based on the style and content layers
    style_output = [vgg.get_layer(name).output for name in style_layers]
    content_output = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_output + content_output 
    
    # Build the model
    return models.Model(vgg.input, model_outputs)

    
# Content Loss Function
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target)) 


# Generate the Gram Matrix
def gram_matrix(input_tensor):
    # Generate image channels. If the input tensor is a 3D array of size Nh x Nw x Nc, reshape
    # it to a 2D array of Nc x (Nh*Nw)
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram/tf.cast(n, tf.float32)


# Style Loss Function
def get_style_loss(base_style, gram_target):
    """ Expects 2 images with dimension h, w, c"""
    # Height, width, num  filters of each layer. The loss is scaled at a given layer by the size        
    # of the feature map and the number of filters 
    
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Used to pass the content and style images through
def get_feature_representations(model, content_path, style_path):
    """
    Helper function to compute the feature representations for the content and style
    

    Parameters
    ----------
    model : The model being use, ie VGG19
    content_path : Path to the content image
    style_path : Path to the style image

    Returns
    -------
    Content and Style features

    """    
    # Load image
    content_img = Image.load_process_img(content_path)
    style_img = Image.load_process_img(style_path)
    
    # Batch compute the features
    style_output = model(style_img)
    content_output = model(content_img)
    
    # Retreive the feature representations of teh style and content from the model
    style_features = [style_layer[0] for style_layer in style_output[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_output[num_style_layers:]]
    return style_features, content_features

# Total Loss
def compute_loss(model, loss_weights, init_img, gram_style_features, content_features):
    """
    This function computes the total loss

    Parameters
    ----------
    model : Model that will allow for access to the intermediate layers
    loss_weights : The weights of each contribution of each seperate loss function.
    (style weight, content weight, and total variation weight)
    init_img : The intial base image that is being updated with the optimization process. The gradient is applied WRT 
    the loss that is calculated to this image
    gram_style_features : Precomputed gram matrices that correspond to the defined style layers of interest
    content_features : Precomputed outputs from the defined content layers of interest

    Returns
    -------
    Total loss, style loss, and total variational loss

    """
    style_weight, content_weight = loss_weights
    
    # Feed the initial image to the mode This will provide the content and style representations at the desired layers
    model_outputs = model(init_img)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0
    
    # Accumulate content losses from all the layers
    
    weight_per_style_layer = 1.0/float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer*get_style_loss(comb_style[0], target_style)
        
    weight_per_content_layer = 0.2/float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer*get_content_loss(comb_content[0], target_content)
        
    style_score *= style_weight
    content_score *= content_weight
    
    # Retreive total loss
    loss = style_score + content_score
    return loss, style_score, content_score

# Computes the gradients
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_img']), all_loss

def run_style_transfer(content_path, style_path, num_iterations = 1000, content_weight = 5e0, style_weight = 5e2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
        
    # Retreive the style and content feature representations from the specified int layers
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
        
    # Set initial image
    init_img = Image.load_process_img(content_path)
    init_img = tf.Variable(init_img, dtype = tf.float32)
    
    # Create the optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = 1e1, beta_1 = 0.9, beta_2=0.999, epsilon = 1e-8)
    #opt = tf.keras.optimizers.Adam(learning_rate = 5, beta_1 = 0.99, epsilon = 1e-1)
    
    # For displaying intermediate images
    iter_count = 1
    
    # Store the best result
    best_loss, best_img = float('inf'), None
    
    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_img': init_img,
        'gram_style_features': gram_style_features,
        'content_features': content_features}
    
    # For display
    num_rows = 2
    num_cols= 5
    display_interval = num_iterations/(num_rows*num_cols)
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_img)])
        clipped = tf.clip_by_value(init_img, min_vals, max_vals)
        init_img.assign(clipped)
        
        if loss < best_loss:
            # Update the best loss and image from total loss
            best_loss = loss
            best_img = Image.deprocess_img(init_img.numpy())
            
        if i % display_interval == 0:
            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_img.numpy()
            plot_img = Image.deprocess_img(plot_img)
            
            path = 'output/output_' + str(i) + '.jpg'
            
            Image.save_results(plot_img, path)
            imgs.append(plot_img)
            
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e}, '
                  'Style Loss: {:.4e}, '
                  'Content Loss: {:.4e}'
                  .format(loss, style_score, content_score))
    
    return best_img, best_loss