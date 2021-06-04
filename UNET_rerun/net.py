from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU


def UNET_tanh(num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))

    encoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(inputs)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("selu")(encoder1) 

    encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder1)
    encoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("selu")(encoder2) 

    encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder2)
    encoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("selu")(encoder3) 

    encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder3)
    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("selu")(encoder4)

    encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder4)
    encoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("selu")(encoder5)

    encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder5)
    encoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder6)
    encoder6 = layers.BatchNormalization()(encoder6)
    encoder6 = layers.Activation("selu")(encoder6)

    encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder6)
    encoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder7)
    encoder7 = layers.BatchNormalization()(encoder7)
    encoder7 = layers.Activation("selu")(encoder7)

    bottom = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder7)
    bottom = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(bottom)
    bottom = layers.BatchNormalization()(bottom)
    bottom = layers.Activation("selu")(bottom) 
    bottom = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(bottom)

    decoder7 = layers.Concatenate()([encoder7,bottom])
    decoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder7)
    decoder7 = layers.BatchNormalization()(decoder7)
    decoder7 = layers.Activation("selu")(decoder7) 
    decoder7 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder7)

    decoder6 = layers.Concatenate()([encoder6,decoder7])
    decoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder6)
    decoder6 = layers.BatchNormalization()(decoder6)
    decoder6 = layers.Activation("selu")(decoder6) 
    decoder6 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder6)

    decoder5 = layers.Concatenate()([encoder5,decoder6])
    decoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder5)
    decoder5 = layers.BatchNormalization()(decoder5)
    decoder5 = layers.Activation("selu")(decoder5) 
    decoder5 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder5)
    
    decoder4 = layers.Concatenate()([encoder4,decoder5])
    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("selu")(decoder4) 
    decoder4 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder4)

    decoder3 = layers.Concatenate()([encoder3,decoder4])
    decoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("selu")(decoder3) 
    decoder3 = layers.Conv2DTranspose(filters = 256,kernel_size=2,strides = 2, padding= "same")(decoder3)

    decoder2 = layers.Concatenate()([encoder2,decoder3])
    decoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("selu")(decoder2) 
    decoder2 = layers.Conv2DTranspose(filters = 128,kernel_size=2,strides = 2, padding= "same")(decoder2)  

    decoder1 = layers.Concatenate()([encoder1,decoder2])
    decoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("selu")(decoder1) 

    outputs = layers.Conv2D(num_classes, kernel_size = 1, activation="tanh", padding="same")(decoder1)

    model = keras.Model(inputs, outputs)
    return model

def UNET_tanh(num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))

    encoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(inputs)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("relu")(encoder1) 

    encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder1)
    encoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("relu")(encoder2) 

    encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder2)
    encoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("relu")(encoder3) 

    encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder3)
    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("relu")(encoder4)

    encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder4)
    encoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("relu")(encoder5)

    encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder5)
    encoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder6)
    encoder6 = layers.BatchNormalization()(encoder6)
    encoder6 = layers.Activation("relu")(encoder6)

    encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder6)
    encoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder7)
    encoder7 = layers.BatchNormalization()(encoder7)
    encoder7 = layers.Activation("relu")(encoder7)

    bottom = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder7)
    bottom = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(bottom)
    bottom = layers.BatchNormalization()(bottom)
    bottom = layers.Activation("relu")(bottom) 
    bottom = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(bottom)

    decoder7 = layers.Concatenate()([encoder7,bottom])
    decoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder7)
    decoder7 = layers.BatchNormalization()(decoder7)
    decoder7 = layers.Activation("relu")(decoder7) 
    decoder7 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder7)

    decoder6 = layers.Concatenate()([encoder6,decoder7])
    decoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder6)
    decoder6 = layers.BatchNormalization()(decoder6)
    decoder6 = layers.Activation("relu")(decoder6) 
    decoder6 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder6)

    decoder5 = layers.Concatenate()([encoder5,decoder6])
    decoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder5)
    decoder5 = layers.BatchNormalization()(decoder5)
    decoder5 = layers.Activation("relu")(decoder5) 
    decoder5 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder5)
    
    decoder4 = layers.Concatenate()([encoder4,decoder5])
    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("relu")(decoder4) 
    decoder4 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder4)

    decoder3 = layers.Concatenate()([encoder3,decoder4])
    decoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("relu")(decoder3) 
    decoder3 = layers.Conv2DTranspose(filters = 256,kernel_size=2,strides = 2, padding= "same")(decoder3)

    decoder2 = layers.Concatenate()([encoder2,decoder3])
    decoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("relu")(decoder2) 
    decoder2 = layers.Conv2DTranspose(filters = 128,kernel_size=2,strides = 2, padding= "same")(decoder2)  

    decoder1 = layers.Concatenate()([encoder1,decoder2])
    decoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("relu")(decoder1) 

    outputs = layers.Conv2D(num_classes, kernel_size = 1, activation="sigmoid", padding="same")(decoder1)

    model = keras.Model(inputs, outputs)
    return model