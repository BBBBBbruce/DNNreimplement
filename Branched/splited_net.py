from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

filter_1 = [32,64,128,128,128,128,256,256,256,128,128,128,128,64,32]
filter_2 = [64,128,128,256,256,256,256,256,256,256,256,256,128,128,64]
filters = [64,128,128,128,128,256 ,512,512,512, 256,128,128,128,128,64]

def net_split(filters,num_class):
    inputs = keras.Input(shape=(256,256) + (3,))

    gf       = layers.AveragePooling2D(inputs.shape[1],inputs.shape[1])(inputs)
    gf       = layers.Dense(filters[0])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder1 = layers.Conv2D(filters = filters[0], kernel_size = 4, padding="same")(inputs)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("selu")(encoder1) 

    encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder1)
    gfdown   = layers.AveragePooling2D(encoder2.shape[1],encoder2.shape[1])(encoder2)
    gfup     = layers.Dense(filters[0])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[1])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder2 = layers.Add()([encoder2,gfup])
    encoder2 = LeakyReLU()(encoder2)
    encoder2 = layers.Conv2D(filters = filters[1], kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("selu")(encoder2) 

    encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder2)
    gfdown   = layers.AveragePooling2D(encoder3.shape[1],encoder3.shape[1])(encoder3)
    gfup     = layers.Dense(filters[1])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[2])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder3 = layers.Add()([encoder3,gfup])
    encoder3 = LeakyReLU()(encoder3)
    encoder3 = layers.Conv2D(filters = filters[2], kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("selu")(encoder3) 

    encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder3)
    gfdown   = layers.AveragePooling2D(encoder4.shape[1],encoder4.shape[1])(encoder4)
    gfup     = layers.Dense(filters[2])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[3])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder4 = layers.Add()([encoder4,gfup])
    encoder4 = LeakyReLU()(encoder4)
    encoder4 = layers.Conv2D(filters = filters[3], kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("selu")(encoder4)

    encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder4)
    gfdown   = layers.AveragePooling2D(encoder5.shape[1],encoder4.shape[1])(encoder5)
    gfup     = layers.Dense(filters[3])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[4])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder5 = layers.Add()([encoder5,gfup])
    encoder5 = LeakyReLU()(encoder5)
    encoder5 = layers.Conv2D(filters = filters[4], kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("selu")(encoder5)

    encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder5)
    gfdown   = layers.AveragePooling2D(encoder6.shape[1],encoder6.shape[1])(encoder6)
    gfup     = layers.Dense(filters[4])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[5])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder6 = layers.Add()([encoder6,gfup])
    encoder6 = LeakyReLU()(encoder6)    
    encoder6 = layers.Conv2D(filters = filters[5], kernel_size = 4, padding="same")(encoder6)
    encoder6 = layers.BatchNormalization()(encoder6)
    encoder6 = layers.Activation("selu")(encoder6)

    encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder6)
    gfdown   = layers.AveragePooling2D(encoder7.shape[1],encoder7.shape[1])(encoder7)
    gfup     = layers.Dense(filters[5])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[6])(gf)
    gf       = layers.Activation('selu')(gf)
    encoder7 = layers.Add()([encoder7,gfup])
    encoder7 = LeakyReLU()(encoder7)    
    encoder7 = layers.Conv2D(filters = filters[6], kernel_size = 4, padding="same")(encoder7)
    encoder7 = layers.BatchNormalization()(encoder7)
    encoder7 = layers.Activation("selu")(encoder7)

    bottom   = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder7)
    gfdown   = layers.AveragePooling2D(bottom.shape[1],bottom.shape[1])(bottom)
    gfup     = layers.Dense(filters[6])(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[7])(gf)
    gf       = layers.Activation('selu')(gf)
    bottom   = layers.Add()([bottom,gfup])
    bottom   = LeakyReLU()(bottom)    
    bottom   = layers.Conv2D(filters = filters[7], kernel_size = 4, padding="same")(bottom)
    bottom   = layers.BatchNormalization()(bottom)
    bottom   = layers.Activation("selu")(bottom) 
    bottom   = layers.Conv2DTranspose(filters = filters[8],kernel_size=2,strides = 2, padding= "same")(bottom)

    decoder7 = layers.Concatenate()([encoder7,bottom])
    gfdown   = layers.AveragePooling2D(decoder7.shape[1],decoder7.shape[1])(decoder7)
    gfup     = layers.Dense(filters[8]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[8])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder7 = layers.Add()([decoder7,gfup])
    decoder7 = LeakyReLU()(decoder7)    
    decoder7 = layers.Conv2D(filters = filters[8], kernel_size = 4, padding="same")(decoder7)
    decoder7 = layers.BatchNormalization()(decoder7)
    decoder7 = layers.Activation("selu")(decoder7) 
    decoder7 = layers.Conv2DTranspose(filters = filters[9],kernel_size=2,strides = 2, padding= "same")(decoder7)

    decoder6 = layers.Concatenate()([encoder6,decoder7])
    gfdown   = layers.AveragePooling2D(decoder6.shape[1],decoder6.shape[1])(decoder6)
    gfup     = layers.Dense(filters[9]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[9])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder6 = layers.Add()([decoder6,gfup])
    decoder6 = LeakyReLU()(decoder6)    
    decoder6 = layers.Conv2D(filters = filters[9], kernel_size = 4, padding="same")(decoder6)
    decoder6 = layers.BatchNormalization()(decoder6)
    decoder6 = layers.Activation("selu")(decoder6) 
    decoder6 = layers.Conv2DTranspose(filters = filters[10],kernel_size=2,strides = 2, padding= "same")(decoder6)

    decoder5 = layers.Concatenate()([encoder5,decoder6])
    gfdown   = layers.AveragePooling2D(decoder5.shape[1],decoder5.shape[1])(decoder5)
    gfup     = layers.Dense(filters[10]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[10])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder5 = layers.Add()([decoder5,gfup])
    decoder5 = LeakyReLU()(decoder5) 
    decoder5 = layers.Conv2D(filters = filters[10], kernel_size = 4, padding="same")(decoder5)
    decoder5 = layers.BatchNormalization()(decoder5)
    decoder5 = layers.Activation("selu")(decoder5) 
    decoder5 = layers.Conv2DTranspose(filters = filters[11],kernel_size=2,strides = 2, padding= "same")(decoder5)
    
    decoder4 = layers.Concatenate()([encoder4,decoder5])
    gfdown   = layers.AveragePooling2D(decoder4.shape[1],decoder4.shape[1])(decoder4)
    gfup     = layers.Dense(filters[11]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[11])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder4 = layers.Add()([decoder4,gfup])
    decoder4 = LeakyReLU()(decoder4) 
    decoder4 = layers.Conv2D(filters = filters[11], kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("selu")(decoder4) 
    decoder4 = layers.Conv2DTranspose(filters = filters[12],kernel_size=2,strides = 2, padding= "same")(decoder4)

    decoder3 = layers.Concatenate()([encoder3,decoder4])
    gfdown   = layers.AveragePooling2D(decoder3.shape[1],decoder3.shape[1])(decoder3)
    gfup     = layers.Dense(filters[12]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[12])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder3 = layers.Add()([decoder3,gfup])
    decoder3 = LeakyReLU()(decoder3) 
    decoder3 = layers.Conv2D(filters = filters[12], kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("selu")(decoder3) 
    decoder3 = layers.Conv2DTranspose(filters = filters[13],kernel_size=2,strides = 2, padding= "same")(decoder3)

    decoder2 = layers.Concatenate()([encoder2,decoder3])
    gfdown   = layers.AveragePooling2D(decoder2.shape[1],decoder2.shape[1])(decoder2)
    gfup     = layers.Dense(filters[13]*2)(gf)
    gf       = layers.Concatenate()([gf,gfdown])
    gf       = layers.Dense(filters[13])(gf)
    gf       = layers.Activation('selu')(gf)
    decoder2 = layers.Add()([decoder2,gfup])
    decoder2 = LeakyReLU()(decoder2) 
    decoder2 = layers.Conv2D(filters = filters[13], kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("selu")(decoder2) 
    decoder2 = layers.Conv2DTranspose(filters = filters[14],kernel_size=2,strides = 2, padding= "same")(decoder2)  

    decoder1 = layers.Concatenate()([encoder1,decoder2])
    gfup     = layers.Dense(filters[14]*2)(gf)
    decoder1 = layers.Add()([decoder1,gfup])
    decoder1 = LeakyReLU()(decoder1) 
    decoder1 = layers.Conv2D(filters = filters[14], kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("selu")(decoder1) 

    outputs = layers.Conv2D(num_class, kernel_size = 1, activation="tanh", padding="same")(decoder1)

    model = keras.Model(inputs, outputs)
    return model


