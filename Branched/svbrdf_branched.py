from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

from blocks import SingleBranch

def UNET_branched():
    inputs         = keras.Input(shape=(256,256) + (3,))

    albedo         = SingleBranch()(inputs)
    albedo_out     = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(albedo)
    specular       = SingleBranch()(inputs)
    specular_out   = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(specular)
    normal         = SingleBranch()(inputs)
    normal_out     = layers.Conv2D(2, kernel_size = 1, activation="tanh", padding="same")(normal)
    roughness      = SingleBranch()(inputs)
    roughness_out  = layers.Conv2D(1, kernel_size = 1, activation="tanh", padding="same")(roughness)

    outputs = layers.Concatenate()([albedo_out,specular_out,normal_out,roughness_out])
    model = keras.Model(inputs, outputs)
    return model



model = UNET_branched()
model.summary()