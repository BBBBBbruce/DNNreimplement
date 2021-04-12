import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional import UpSampling1D

def svbrdf(num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))
    x = layers.Layer()(inputs)
    previous_block_activation = x  # Set aside residual


    downfilters = np.array([128,256,512,512,512,512,512,512])
    Upfilters = np.flip(np.copy(downfilters))

    for filters in downfilters:
        
        x = layers.LeakyReLU()(x)
        x = layers.SeparableConv2D(filters, 4,2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        # Project residual

        residual = layers.Conv2D(filters, 4,2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in Upfilters:
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(filters, 4, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)

        
        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 4, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = svbrdf(9)
model.summary()
