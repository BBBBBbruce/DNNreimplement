import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers

from GGXrenderer import GGXtf

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

def rendering_loss(mgt, mif):

    return l1_loss(GGXtf(mgt),GGXtf(mif))


def SVBRDF(num_classes):
    #=============== first layer ==================

    inputs = keras.Input(shape=(256,256) + (3,))
    x = layers.LeakyReLU()(inputs)
    GF = layers.AveragePooling2D(x.shape[1])(x)
    GF = layers.Dense(128)(GF)
    GF = layers.Activation('selu')(GF)
    x = layers.SeparableConv2D(128,4, 2, padding="same")(x)
    #previous_block_activation = x  # Set aside residual

    #========== define filters for unet ===================

    downfilters = np.array([128,256,512,512,512,512,512,512])
    Upfilters = np.flip(np.copy(downfilters))
    downfilters = np.delete(downfilters,0)
    #print(downfilters)
    prefilter = 128

    #===================== upsampling =======================

    for filters in downfilters:
        #print(x.shape)
        #print(filters)
        GFdown = layers.AveragePooling2D(x.shape[1])(x)
        GFup   = layers.Dense(prefilter)(x)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.SeparableConv2D(filters, 4,2, padding="same")(x)
        prefilter = filters
        #x = layers.BatchNormalization()(x)
        # Project residual

        #residual = layers.Conv2D(filters, 4,2, padding="same")(previous_block_activation)
        #x = layers.add([x, residual])  # Add back residual
        #previous_block_activation = x  # Set aside next residual

    #====================== downsampling ============================

    for filters in Upfilters:

        GFdown = layers.AveragePooling2D(x.shape[1])(x)
        GFup   = layers.Dense(prefilter)(x)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(filters, 4,2, padding="same")(x)
        prefilter = filters
        
        # Project residual
        #residual = layers.UpSampling2D(2)(previous_block_activation)
        #residual = layers.Conv2D(filters, 4, padding="same")(residual)
        #x = layers.add([x, residual])  # Add back residual
        #previous_block_activation = x  # Set aside next residual

    #====================== last connection =====================

    GFup   = layers.Dense(prefilter)(x)
    x = layers.Add()([x,GFup])
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model

#keras.backend.clear_session()
#svbrdf = SVBRDF(9)
#svbrdf.summary()
