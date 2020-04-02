# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:44:30 2020

@author: ealegre
"""
import numpy as np
import scipy.io
import scipy.misc
import imageio
import tensorflow as tf
from PIL import Image
import os
from tensorflow.python.keras.preprocessing import image as tf_image


##############################
#####     PARAMETERS     #####
##############################

# Style Image Input Path, a
STYLE_IMAGE = 'input/style/starry.jpg'

# Content Image Input Path. p
CONTENT_IMAGE = 'input/content/original.jpg'

# Output folder for the style transfered images
OUTPUT_PATH = 'output'

# Output Image Constraints, extracted from the input image to match
IMAGE_WIDTH = Image.open(CONTENT_IMAGE).size[0]
IMAGE_HEIGHT =Image.open(CONTENT_IMAGE).size[1]
COLOUR_CHANNELS = 3

ITERATIONS  = 5000

# Define the noise ratio to be used when generating the white noise image. 
# The value is the percentage ofweight of the noise for intermixing with the content image
NOISE_RATIO = 0.6

# Total Loss Alpha value. Corresponds directly with the content loss
ALPHA = 5

# Total Loss Beta value. Corresponds directly with the style loss
BETA = 500

# This code requires the VGG19 model to be in the directory. For ease, this code could have used Keras but decided to 
# try a blind implementation due to v1 of this code using Keras cause confusion. The VGG19 model can be downladed
# https://drive.google.com/file/d/0B8QJdgMvQDrVU2cyZjFKU1RrLUU/view

VGG19 = 'imagenet-vgg-verydeep-19.mat'

# These mean values are used to subtract from the input to the VGG model. This is the mean
# that was used to train the VGG, DO NOT CHANGE THEM
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])

##################################
#####     IMAGE FUNCTIONS    #####
##################################

def white_noise_image(content_image, noise_ratio=NOISE_RATIO):
    """
    

    Parameters
    ----------
    content_image : Image Array
        Content Image
    noise_ratio : scalar
        The default is NOISE_RATIO.

    Returns
    -------
    White noise image

    """
    
    noise_img = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOUR_CHANNELS)).astype('float32')
    
    generated_image = noise_img * noise_ratio + content_image * (1 - noise_ratio)
    return generated_image

def load_image(path):
    # img = imageio.imread(path)
    max_width_dim = IMAGE_WIDTH
    max_height_dim = IMAGE_HEIGHT
    # img_size = max(img.shape)
    # scale = max_dim/img_size
    # # Resize the image for utilization for the CNN input. Basically, size stays the same but adding an extra dimension
    # img = np.reshape(img, ((1,) + img.shape))
    img = Image.open(path)
    # Possibly try Image.BILINEAR instead?
    img = img.resize((round(img.size[0]*(max_width_dim/img.size[0])), round(img.size[1]*(max_height_dim/img.size[1]))), Image.BILINEAR)
    img = tf_image.img_to_array(img)
    
    # Input to the VGG model expects the mean to be subtracted
    img = img - MEAN_VALUES
    return img

def save_image(path, img):
    # Output should add back to the mean
    img = img + MEAN_VALUES
    
    # Since we are saving the image, we don't need the first dimension anymore
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)


###############################
#####     VGG19 MODEL     #####
###############################

def get_vgg19_model(model_path):
    """
    Takes only the convolutional layer weights and wraps using TF
    As per Gatys et al's paper, the layers used will be Conv2d, ReLU for the threshold, and Average for pooling
    All the layers are defined below but when the actual style and content layer weights are defined, it will reflect
    what is stated on the paper

    Parameters
    ----------
    model_path : VGG19
        

    Returns
    -------
    Model for the purpose of transfering the style to the content image

    """
    
    vgg = scipy.io.loadmat(model_path)
    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        """
        

        Parameters
        ----------
        layer : Current Layer
        expected_layer_name : Expected Layer Name, will overwrite 

        Returns
        -------
        weights and biases from the VGG19 model for a given layer

        """
        weight = vgg_layers[0][layer][0][0][0][0][0]
        bias = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        
        assert (layer_name == expected_layer_name)
        return weight, bias
    
    def _relu(conv2d_layer):
        """
        

        Parameters
        ----------
        conv2d_layer: Convolutional Layer

        Returns
        -------
        ReLU function wrapped over a TF layer.

        """
        
        return tf.nn.relu(conv2d_layer)
    
    def _conv2d(prev_layer, layer, layer_name):
        """
        

        Parameters
        ----------
        pre_layer : Previous Layer
        layer : Current Layer
        layer_name : Layer Name

        Returns
        -------
        Conv2D layer using the weights and biases from VGG model at 'layer'

        """
        
        weight, bias = _weights(layer, layer_name)
        weight = tf.constant(weight)
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return tf.nn.conv2d(prev_layer, filters=weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    
    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        

        Parameters
        ----------
        pre_layer : Previous Layer
        layer : Current Layer
        layer_name : Layer Name

        Returns
        -------
        Conv2D layer using the weights and biases from VGG model at 'layer'

        """
        
        return _relu(_conv2d(prev_layer, layer, layer_name))
    
    def _avgpooling(prev_layer):
        """
        Performs average pooling 

        Parameters
        ----------
        prev_layer : Previous layer
        
        Returns
        -------
        Average Pooling layer

        """
        
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    graph_model = {}
    graph_model['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOUR_CHANNELS)), dtype = 'float32')
    graph_model['conv1_1'] = _conv2d_relu(graph_model['input'], 0, 'conv1_1')
    # Recall that in between each layer is a relu, thus Layer 1 is a ReLU layer
    graph_model['conv1_2'] = _conv2d_relu(graph_model['conv1_1'], 2, 'conv1_2')
    graph_model['avgpool1'] = _avgpooling(graph_model['conv1_2'])
    # ReLU Layer 4
    graph_model['conv2_1'] = _conv2d_relu(graph_model['avgpool1'], 5, 'conv2_1')
    # ReLU Layer 6
    graph_model['conv2_2'] = _conv2d_relu(graph_model['conv2_1'], 7, 'conv2_2')
    graph_model['avgpool2'] = _avgpooling(graph_model['conv2_2'])
    # ReLU Layer 9
    graph_model['conv3_1'] = _conv2d_relu(graph_model['avgpool2'], 10, 'conv3_1')
    # ReLU Layer 11
    graph_model['conv3_2'] = _conv2d_relu(graph_model['conv3_1'], 12, 'conv3_2')
    # ReLU Layer 13
    graph_model['conv3_3'] = _conv2d_relu(graph_model['conv3_2'], 14, 'conv3_3')
    # ReLU Layer 15
    graph_model['conv3_4'] = _conv2d_relu(graph_model['conv3_3'], 16, 'conv3_4')    
    graph_model['avgpool3'] = _avgpooling(graph_model['conv3_4'])
    # ReLU Layer 18
    graph_model['conv4_1'] = _conv2d_relu(graph_model['avgpool3'], 19, 'conv4_1')
    # ReLU Layer 20
    graph_model['conv4_2'] = _conv2d_relu(graph_model['conv4_1'], 21, 'conv4_2')
    # ReLU Layer 22
    graph_model['conv4_3'] = _conv2d_relu(graph_model['conv4_2'], 23, 'conv4_3')
    # ReLU Layer 24
    graph_model['conv4_4'] = _conv2d_relu(graph_model['conv4_3'], 25, 'conv4_4')    
    graph_model['avgpool4'] = _avgpooling(graph_model['conv4_4'])
    # ReLU Layer 27
    graph_model['conv5_1'] = _conv2d_relu(graph_model['avgpool4'], 28, 'conv5_1')
    # ReLU Layer 29
    graph_model['conv5_2'] = _conv2d_relu(graph_model['conv5_1'], 30, 'conv5_2')
    # ReLU Layer 31
    graph_model['conv5_3'] = _conv2d_relu(graph_model['conv5_2'], 32, 'conv5_3')
    # ReLU Layer 33
    graph_model['conv5_4'] = _conv2d_relu(graph_model['conv5_3'], 34, 'conv5_4')    
    graph_model['avgpool5'] = _avgpooling(graph_model['conv5_4'])    
    
    return graph_model
    

#########################################
#####     CONTENT LOSS FUNCTION     #####
#########################################
    
def content_loss_function(sess, model):
    """
    Equation 1 of the paper

    Parameters
    ----------
    sess : tf.compat.v1.Session() 
        Tensorflow session
    model : VGG19, with layer specified
        conv4_2

    Returns
    -------
    Content loss

    """
    
    def _content_loss(p, x):
        L_content = 0.5 * tf.reduce_sum(tf.square(x - p))
        return L_content
    
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

#####################################################
#####     STYLE LOSS FUNCTION & GRAM MATRIX     #####
#####################################################

def style_loss_function(sess, model):
    """
    Equation 5 of the paper

    Parameters
    ----------
     sess : tf.compat.v1.Session() 
        Tensorflow session
    model : VGG19, with layer specified
        conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

    Returns
    -------
    Style Loss

    """
    
    def _gram_matrix(F, N, M):
        # Equation 3 of the paper
        
        Ft = tf.reshape(F, (M, N))
        gram = tf.matmul(tf.transpose(Ft), Ft)
        
        return gram
    
    def _style_loss(a, x):
        # N = number of filters at layer l
        N = a.shape[3]
        
        # M is the height times the width of the feature map at layer l
        M = a.shape[1] * a.shape[2]
        
        # A is the style representation of the original image at layer l
        A = _gram_matrix(a, N, M)
        
        # G is the style representation of the generated white noise image at layer l
        G = _gram_matrix(x, N, M)
        
        E_l = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.square(G - A))
        return E_l
    
    # Specified layers that the style representation is passed through. Technically,
    # the weights should all be 0.2 but since I want more of the style to come through, I've
    # specified that the higher level features should be weighed more to make it so
    
    style_layers =[
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0)
        ]
    
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in style_layers]
    W = [W for _, W in style_layers]
    L_style = sum([W[1] * E[1] for l in range(len(style_layers))])
    return L_style


#################################
#####     MAIN FUNCTION     #####
#################################
    
if __name__ == '__main__': 
    with tf.compat.v1.Session() as sess:
        # Load the images and model
        content_img = load_image(CONTENT_IMAGE)
        style_img = load_image(STYLE_IMAGE)
        model = get_vgg19_model(VGG19)
        
        generated_img = white_noise_image(content_img)
        
        sess.run(tf.compat.v1.initialize_all_variables())
        # Construct losses
        sess.run(model['input'].assign(content_img))
        content_loss = content_loss_function(sess, model)
        
        sess.run(model['input'].assign(style_img))
        style_loss = style_loss_function(sess, model)
        
        # Equation 7 of the paper  
        L_total = ALPHA * content_loss + BETA * style_loss
        
        opt = tf.comapt.v1.train.AdamOptimizer(learning_rate = 2.0, epsilon = 1e-8)
        train_step = opt.minimize(L_total)
        
        sess.run(tf.compat.v1.initialize_all_variables())
        sess.run(model['input'].assign(generated_img))
        
        for iteration in range(ITERATIONS):
            sess.run(train_step)
            if iteration%100 == 0:
            # Prints every 100 iterations
                synthesized_img = sess.run(model['input'])
                print('Iteration %d' % (iteration))
                print('Sum: ', sess.run(tf.reduce_sum(synthesized_img)))
                print('Cost: ', sess.run(L_total))
                
                if not os.path.exists(OUTPUT_PATH):
                    os.mkdir(OUTPUT_PATH)
                    
                file_name = 'output%d.jpg' % (iteration)
                save_image(file_name, synthesized_img)
        