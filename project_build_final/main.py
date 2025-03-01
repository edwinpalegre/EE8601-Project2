# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:16:38 2020

@author: ealegre
"""

# DEPENDENCIES, VARIABLES, AND PATHS

# Libraries
import tensorflow as tf
from tensorflow.python.keras import models
import os
import glob
import image_utils as IU

# Define paths
content_path = 'input/content/elon.jpg'
style_path = 'input/style/ironman.jpg'
init_path = 'input/init/512.jpg'
lena = 'input/content/lena_test.png'
lion = 'input/content/lion.jpg'
dog = 'input/content/dog.jpg'

# Load the style and content images
style_img = IU.load_img(style_path)
content_img = IU.load_img(content_path)
generated_img = tf.Variable(content_img)

# Define which layers are to be used for this model. These layers are defined in Section 
# 3 of Gatys' paper
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# MODEL UTILITIES - FUNCTION 1 - VGG19 LAYERS

def get_vggLayers(style_or_content_layers):
    """
    Creates the model with intermediate layer access
    
    This function will load in the VGG19 model used in Gatys' paper, with access to the 
    intermediate layers. A new model will be generated by using these layers that will take an 
    input image and return an output from the intermediate layers from the VGG19 model

    Returns
    -------
    A Keras model that takes inputs and outputs of the style and content intermediate layers

    """
    # Instantiate the VGG19 model. We are not including the fully connected layers, instantiate the weight based off ImageNet training
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # VGG19 model is already trained off ImageNet, set trainable to False
    vgg.trainable = False
    
    # All we are doing is clipping the model accrodingly. Just set the input of the model to be the same as its original
    # The outputs are going to be set to the layers specified in either the style or content layers, this will
    # be set once you've passed one of them into the function as an argument
    vgg_input = [vgg.input]
    vgg_output = [vgg.get_layer(name).output for name in style_or_content_layers]
    model = tf.keras.Model(vgg_input, vgg_output)

    return model

# MODEL UTILITIES - FUNCTION 2 - THE GRAM MATRIX

def gram_matrix(input_tensor):
  """
  Generates the Gram matrix representation of the style features. This function takes an input tensor and
  will apply the proper steps to generate the Gram matrix

  Returns
  -------
  A tensor object Gram matrix representation
  
  """
  # Equation (3)

  # Generate image channels. If the input tensor is a 3D array of size Nh x Nw x Nc, reshape
  # it to a 2D array of Nc x (Nh*Nw). The shape[-1] takes the last element in the shape
  # characterstic, that being the number of channels. This will be our second dimension
  channels = int(input_tensor.shape[-1])

  # Reshape the tensor into a 2D matrix to prepare for Gram matrix calculation by multiplying
  # all of the dimensions except the last one (which is what the -1 represents) together
  Fl_ik = tf.reshape(input_tensor, [-1, channels])

  # Transpose the new 2D matrix
  Fl_jk = tf.transpose(Fl_ik)

  # Find all the elements in the new array (Nw*Nh)
  n = tf.shape(Fl_ik)[0]

  # Perform the Gram matrix calculation
  gram = tf.matmul(Fl_jk, Fl_ik)/tf.cast(n, tf.float32)

  # Generate the Gram matrix as a tensor for use in our model
  gram_tensor = gram[tf.newaxis, :]
  return gram_tensor

# MODEL UTILITIES - FUNCTION 3 - STYLE & CONTENT MODEL CLASS

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = get_vggLayers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False
  
  def call(self, inputs):
    # Expects a float input between [0, 1] 
    inputs = inputs * 255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}

    return{'content':content_dict, 'style':style_dict}


feature_extractor = StyleContentModel(style_layers, content_layers)

# Load the style and content images
style_img = IU.load_img(style_path)
content_img = IU.load_img(content_path)

# Extract the features from the style and content images
style_features = feature_extractor(style_img)['style']
content_features = feature_extractor(content_img)['content']

# The Tensorflow variable function initializes our image to be used for gradient descent. 
# This image has to be the same size and type as the content image, which means that as long as it's loaded in before being 
# called upon as the generated image, it should be fine to use. This is essentially the image that will be optimized.
# In case you want to use another image, you can uncomment the code below
'''
init_img = load_img(content_path)
generated_image = tf.Variable(init_img)
'''
generated_img = tf.Variable(content_img)

# Remember that all of our images are read in as a float32 type, so we have to define the range as [0, 1] to keep it within 255
def clip_range(img):
  return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

# Choose an optimizer. The paper chose L-BFGS but Tensorflow doesn't have that, so we'll use ADAM
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-8)

# Remember that we are trying to optimize the Total Loss function. However, there are values that correspond to the style and content weight
# The Alpha value corresponds to content weight and the Beta corresponds to the style weight. Let's define these
alpha = 1e4
beta = 10

# MODEL UTILITIES - FUNCTION 4 - STYLE, CONTENT, & TOTAL LOSS FUNCTIONS

# Now we define our loss functions
def style_content_loss(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']

  # Equation 5
  # Define the style loss function
  style_loss = tf.add_n([tf.reduce_mean(tf.square(style_outputs[name] - style_features[name])) for name in style_outputs.keys()])
  # Multiply the style loss by the weighted variable to get the weighted style loss
  style_loss *= beta / num_style_layers

  # Equation 1
  # Define the content loss function
  content_loss = tf.add_n([tf.reduce_mean(tf.square(content_outputs[name] - content_features[name])) for name in content_outputs.keys()])
  # Multiply the content loss by the weighted variable to get the weighted content loss
  content_loss *= alpha / num_content_layers
  
  # Equation 7
  # Define the total loss function
  total_loss = style_loss + content_loss
  return total_loss

# State the total variational weight
total_variational_weight= 30

@tf.function()
def train_step(img):
  # Tensorflow's GradientTape function performs automatic differentiation of the input
  with tf.GradientTape() as tape:
    outputs = feature_extractor(img)
    loss = style_content_loss(outputs)
    loss += total_variational_weight * tf.image.total_variation(img)

  # Apply the gradient by passing the loss and the generated image
  grad = tape.gradient(loss, img)

  # The gradient is optimized using the ADAM optimizer we declared earlier. This optimizes the 
  # generated image using the gradient values
  optimizer.apply_gradients([(grad, img)])

  # The image is rewritten, being sure that the values all stay within the viable range of [0, 1]
  img.assign(clip_range(img))
  
import time
from IPython import display
# Start the run time
start = time.time()

epochs = 100
iterations = 100

step = 0
for n in range(epochs):
  for m in range(iterations):
    step += 1
    train_step(generated_img)
    print(".", end=' ')
  display.clear_output(wait=True)
  display.display(IU.deprocess_img(generated_img))
  print("Train step: {}".format(step))

end = time.time()
print("Total Time: {:.1f}".format(end-start))

file_name = 'output/stylized-image.png'
IU.deprocess_img(generated_img).save(file_name)

