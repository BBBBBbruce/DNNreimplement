from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
#
def SVBRDF(num_classes):
    #=============== first layer ==================

    inputs = keras.Input(shape=(256,256) + (3,))
    x = layers.LeakyReLU()(inputs)
    GF = layers.AveragePooling2D((x.shape[1],x.shape[1]))(x)
    GF = layers.Dense(128)(GF)
    GF = layers.Activation('selu')(GF)
    x = layers.SeparableConv2D(128,4, 2, padding="same")(x)
    #previous_block_activation = x  # Set aside residual

    #========== define filters for unet ===================

    downfilters = np.array([128,256,512,512,512,512,512,512])
    Upfilters = np.flip(np.copy(downfilters))
    downfilters = np.delete(downfilters,0)
    prefilter = 128

    #===================== upsampling =======================

    for filters in downfilters:
        #print(x.shape)
        #print(filters)
        GFdown = layers.AveragePooling2D((x.shape[1],x.shape[1]))(x)
        GFup   = layers.Dense(prefilter)(x)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.SeparableConv2D(filters, 4,2, padding="same")(x)
        prefilter = filters

    #====================== downsampling ============================

    for filters in Upfilters:

        GFdown = layers.AveragePooling2D((x.shape[1],x.shape[1]))(x)
        GFup   = layers.Dense(prefilter)(x)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(filters, 4,2, padding="same")(x)
        prefilter = filters

    #====================== last connection =====================

    GFup   = layers.Dense(prefilter)(x)
    x = layers.Add()([x,GFup])
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model

