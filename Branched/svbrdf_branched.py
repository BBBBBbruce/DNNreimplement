from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

from blocks import SingleBranch_u

from blocks_s import SingleBranch_s

def UNET_branched():
    inputs         = keras.Input(shape=(256,256) + (3,))
    #filters = [128,256,512,512,512,512,512,512,512,512,512,512,512,256,128]
    filter_1 = [32,64,128,128,128,128,256,256,256,128,128,128,128,64,32]
    filter_2 = [64,128,128,256,256,256,256,256,256,256,256,256,128,128,64]
    filter_3 = [64,128,128,128,128,256 ,512,512,512, 256,128,128,128,128,64]
    #filter_24 = filter_8 *3

    albedo         = SingleBranch_u(filter_3)(inputs)
    albedo_out     = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(albedo)
    specular       = SingleBranch_u(filter_3)(inputs)
    specular_out   = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(specular)
    normal         = SingleBranch_u(filter_2)(inputs)
    normal_out     = layers.Conv2D(2, kernel_size = 1, activation="tanh", padding="same")(normal)
    roughness      = SingleBranch_u(filter_1)(inputs)
    roughness_out  = layers.Conv2D(1, kernel_size = 1, activation="tanh", padding="same")(roughness)

    outputs = layers.Concatenate()([albedo_out,specular_out,normal_out,roughness_out])
    model = keras.Model(inputs, outputs)
    return model


def svbrdf_branched():
    inputs         = keras.Input(shape=(256,256) + (3,))
    filters = [128,256,512,512,512,512,512,512,512,512,512,512,512,256,128]
    #85,597,440
    filter_1 = [32,64,128,128,128,128,256,256,256,128,128,128,128,64,32]
    filter_2 = [64,128,128,256,256,256,256,256,256,256,256,256,128,128,64]
    filter_3 = [64,128,128,128,128,256 ,512,512,512, 256,128,128,128,128,64]
    albedo         = SingleBranch_s(filter_3)(inputs)
    albedo_out     = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(albedo)
    specular       = SingleBranch_s(filter_3)(inputs)
    specular_out   = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(specular)
    normal         = SingleBranch_s(filter_2)(inputs)
    normal_out     = layers.Conv2D(2, kernel_size = 1, activation="tanh", padding="same")(normal)
    roughness      = SingleBranch_s(filter_1)(inputs)
    roughness_out  = layers.Conv2D(1, kernel_size = 1, activation="tanh", padding="same")(roughness)

    outputs = layers.Concatenate()([albedo_out,specular_out,normal_out,roughness_out])
    model = keras.Model(inputs, outputs)
    return model

model = svbrdf_branched()
model.summary()