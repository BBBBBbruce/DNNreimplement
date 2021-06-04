from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU


filter_6 = [128,128,256,256,512,512 ,512,512,512, 512,512,256,256,128,128]
filter_3 = [64,128,128,128,128,256 ,512,512,512, 256,128,128,128,128,64]

def SVBRDF_partial_branched():
    inputs = keras.Input(shape=(256,256) + (3,))

### just write the complete network, long code, but works. 
### keras subclassing is trash.
### trainable but cannot be loaded.
#a: albedo, s:specular, n: normal, r: roughness

#================
#==== albedo ====
#================

    a_gf       = layers.AveragePooling2D(inputs.shape[1],inputs.shape[1])(inputs)
    a_gf       = layers.Dense(filter_3[0])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder1 = layers.Conv2D(filters = filter_3[0], kernel_size = 4, padding="same")(inputs)
    a_encoder1 = layers.BatchNormalization()(a_encoder1)
    a_encoder1 = layers.Activation("selu")(a_encoder1) 

    a_encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder1)
    a_gfdown   = layers.AveragePooling2D(a_encoder2.shape[1],a_encoder2.shape[1])(a_encoder2)
    a_gfup     = layers.Dense(filter_3[0])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[1])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder2 = layers.Add()([a_encoder2,a_gfup])
    a_encoder2 = LeakyReLU()(a_encoder2)
    a_encoder2 = layers.Conv2D(filters = filter_3[1], kernel_size = 4, padding="same")(a_encoder2)
    a_encoder2 = layers.BatchNormalization()(a_encoder2)
    a_encoder2 = layers.Activation("selu")(a_encoder2) 

    a_encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder2)
    a_gfdown   = layers.AveragePooling2D(a_encoder3.shape[1],a_encoder3.shape[1])(a_encoder3)
    a_gfup     = layers.Dense(filter_3[1])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[2])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder3 = layers.Add()([a_encoder3,a_gfup])
    a_encoder3 = LeakyReLU()(a_encoder3)
    a_encoder3 = layers.Conv2D(filters = filter_3[2], kernel_size = 4, padding="same")(a_encoder3)
    a_encoder3 = layers.BatchNormalization()(a_encoder3)
    a_encoder3 = layers.Activation("selu")(a_encoder3) 

    a_encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder3)
    a_gfdown   = layers.AveragePooling2D(a_encoder4.shape[1],a_encoder4.shape[1])(a_encoder4)
    a_gfup     = layers.Dense(filter_3[2])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[3])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder4 = layers.Add()([a_encoder4,a_gfup])
    a_encoder4 = LeakyReLU()(a_encoder4)
    a_encoder4 = layers.Conv2D(filters = filter_3[3], kernel_size = 4, padding="same")(a_encoder4)
    a_encoder4 = layers.BatchNormalization()(a_encoder4)
    a_encoder4 = layers.Activation("selu")(a_encoder4)

    a_encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder4)
    a_gfdown   = layers.AveragePooling2D(a_encoder5.shape[1],a_encoder4.shape[1])(a_encoder5)
    a_gfup     = layers.Dense(filter_3[3])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[4])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder5 = layers.Add()([a_encoder5,a_gfup])
    a_encoder5 = LeakyReLU()(a_encoder5)
    a_encoder5 = layers.Conv2D(filters = filter_3[4], kernel_size = 4, padding="same")(a_encoder5)
    a_encoder5 = layers.BatchNormalization()(a_encoder5)
    a_encoder5 = layers.Activation("selu")(a_encoder5)

    a_encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder5)
    a_gfdown   = layers.AveragePooling2D(a_encoder6.shape[1],a_encoder6.shape[1])(a_encoder6)
    a_gfup     = layers.Dense(filter_3[4])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[5])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder6 = layers.Add()([a_encoder6,a_gfup])
    a_encoder6 = LeakyReLU()(a_encoder6)    
    a_encoder6 = layers.Conv2D(filters = filter_3[5], kernel_size = 4, padding="same")(a_encoder6)
    a_encoder6 = layers.BatchNormalization()(a_encoder6)
    a_encoder6 = layers.Activation("selu")(a_encoder6)

    a_encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder6)
    a_gfdown   = layers.AveragePooling2D(a_encoder7.shape[1],a_encoder7.shape[1])(a_encoder7)
    a_gfup     = layers.Dense(filter_3[5])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[6])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_encoder7 = layers.Add()([a_encoder7,a_gfup])
    a_encoder7 = LeakyReLU()(a_encoder7)    
    a_encoder7 = layers.Conv2D(filters = filter_3[6], kernel_size = 4, padding="same")(a_encoder7)
    a_encoder7 = layers.BatchNormalization()(a_encoder7)
    a_encoder7 = layers.Activation("selu")(a_encoder7)

    a_bottom   = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(a_encoder7)
    a_gfdown   = layers.AveragePooling2D(a_bottom.shape[1],a_bottom.shape[1])(a_bottom)
    a_gfup     = layers.Dense(filter_3[6])(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[7])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_bottom   = layers.Add()([a_bottom,a_gfup])
    a_bottom   = LeakyReLU()(a_bottom)    
    a_bottom   = layers.Conv2D(filters = filter_3[7], kernel_size = 4, padding="same")(a_bottom)
    a_bottom   = layers.BatchNormalization()(a_bottom)
    a_bottom   = layers.Activation("selu")(a_bottom) 
    a_bottom   = layers.Conv2DTranspose(filters = filter_3[8],kernel_size=2,strides = 2, padding= "same")(a_bottom)

    a_decoder7 = layers.Concatenate()([a_encoder7,a_bottom])
    a_gfdown   = layers.AveragePooling2D(a_decoder7.shape[1],a_decoder7.shape[1])(a_decoder7)
    a_gfup     = layers.Dense(filter_3[8]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[8])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder7 = layers.Add()([a_decoder7,a_gfup])
    a_decoder7 = LeakyReLU()(a_decoder7)    
    a_decoder7 = layers.Conv2D(filters = filter_3[8], kernel_size = 4, padding="same")(a_decoder7)
    a_decoder7 = layers.BatchNormalization()(a_decoder7)
    a_decoder7 = layers.Activation("selu")(a_decoder7) 
    a_decoder7 = layers.Conv2DTranspose(filters = filter_3[9],kernel_size=2,strides = 2, padding= "same")(a_decoder7)

    a_decoder6 = layers.Concatenate()([a_encoder6,a_decoder7])
    a_gfdown   = layers.AveragePooling2D(a_decoder6.shape[1],a_decoder6.shape[1])(a_decoder6)
    a_gfup     = layers.Dense(filter_3[9]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[9])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder6 = layers.Add()([a_decoder6,a_gfup])
    a_decoder6 = LeakyReLU()(a_decoder6)    
    a_decoder6 = layers.Conv2D(filters = filter_3[9], kernel_size = 4, padding="same")(a_decoder6)
    a_decoder6 = layers.BatchNormalization()(a_decoder6)
    a_decoder6 = layers.Activation("selu")(a_decoder6) 
    a_decoder6 = layers.Conv2DTranspose(filters = filter_3[10],kernel_size=2,strides = 2, padding= "same")(a_decoder6)

    a_decoder5 = layers.Concatenate()([a_encoder5,a_decoder6])
    a_gfdown   = layers.AveragePooling2D(a_decoder5.shape[1],a_decoder5.shape[1])(a_decoder5)
    a_gfup     = layers.Dense(filter_3[10]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[10])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder5 = layers.Add()([a_decoder5,a_gfup])
    a_decoder5 = LeakyReLU()(a_decoder5) 
    a_decoder5 = layers.Conv2D(filters = filter_3[10], kernel_size = 4, padding="same")(a_decoder5)
    a_decoder5 = layers.BatchNormalization()(a_decoder5)
    a_decoder5 = layers.Activation("selu")(a_decoder5) 
    a_decoder5 = layers.Conv2DTranspose(filters = filter_3[11],kernel_size=2,strides = 2, padding= "same")(a_decoder5)
    
    a_decoder4 = layers.Concatenate()([a_encoder4,a_decoder5])
    a_gfdown   = layers.AveragePooling2D(a_decoder4.shape[1],a_decoder4.shape[1])(a_decoder4)
    a_gfup     = layers.Dense(filter_3[11]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[11])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder4 = layers.Add()([a_decoder4,a_gfup])
    a_decoder4 = LeakyReLU()(a_decoder4) 
    a_decoder4 = layers.Conv2D(filters = filter_3[11], kernel_size = 4, padding="same")(a_decoder4)
    a_decoder4 = layers.BatchNormalization()(a_decoder4)
    a_decoder4 = layers.Activation("selu")(a_decoder4) 
    a_decoder4 = layers.Conv2DTranspose(filters = filter_3[12],kernel_size=2,strides = 2, padding= "same")(a_decoder4)

    a_decoder3 = layers.Concatenate()([a_encoder3,a_decoder4])
    a_gfdown   = layers.AveragePooling2D(a_decoder3.shape[1],a_decoder3.shape[1])(a_decoder3)
    a_gfup     = layers.Dense(filter_3[12]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[12])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder3 = layers.Add()([a_decoder3,a_gfup])
    a_decoder3 = LeakyReLU()(a_decoder3) 
    a_decoder3 = layers.Conv2D(filters = filter_3[12], kernel_size = 4, padding="same")(a_decoder3)
    a_decoder3 = layers.BatchNormalization()(a_decoder3)
    a_decoder3 = layers.Activation("selu")(a_decoder3) 
    a_decoder3 = layers.Conv2DTranspose(filters = filter_3[13],kernel_size=2,strides = 2, padding= "same")(a_decoder3)

    a_decoder2 = layers.Concatenate()([a_encoder2,a_decoder3])
    a_gfdown   = layers.AveragePooling2D(a_decoder2.shape[1],a_decoder2.shape[1])(a_decoder2)
    a_gfup     = layers.Dense(filter_3[13]*2)(a_gf)
    a_gf       = layers.Concatenate()([a_gf,a_gfdown])
    a_gf       = layers.Dense(filter_3[13])(a_gf)
    a_gf       = layers.Activation('selu')(a_gf)
    a_decoder2 = layers.Add()([a_decoder2,a_gfup])
    a_decoder2 = LeakyReLU()(a_decoder2) 
    a_decoder2 = layers.Conv2D(filters = filter_3[13], kernel_size = 4, padding="same")(a_decoder2)
    a_decoder2 = layers.BatchNormalization()(a_decoder2)
    a_decoder2 = layers.Activation("selu")(a_decoder2) 
    a_decoder2 = layers.Conv2DTranspose(filters = filter_3[14],kernel_size=2,strides = 2, padding= "same")(a_decoder2)  

    a_decoder1 = layers.Concatenate()([a_encoder1,a_decoder2])
    a_gfup     = layers.Dense(filter_3[14]*2)(a_gf)
    a_decoder1 = layers.Add()([a_decoder1,a_gfup])
    a_decoder1 = LeakyReLU()(a_decoder1) 
    a_decoder1 = layers.Conv2D(filters = filter_3[14], kernel_size = 4, padding="same")(a_decoder1)
    a_decoder1 = layers.BatchNormalization()(a_decoder1)
    a_decoder1 = layers.Activation("selu")(a_decoder1) 

    a_outputs = layers.Conv2D(3, kernel_size = 1, activation="tanh", padding="same")(a_decoder1)

#================
#=== specular ===
#================

    s_gf       = layers.AveragePooling2D(inputs.shape[1],inputs.shape[1])(inputs)
    s_gf       = layers.Dense(filter_6[0])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder1 = layers.Conv2D(filters = filter_6[0], kernel_size = 4, padding="same")(inputs)
    s_encoder1 = layers.BatchNormalization()(s_encoder1)
    s_encoder1 = layers.Activation("selu")(s_encoder1) 

    s_encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder1)
    s_gfdown   = layers.AveragePooling2D(s_encoder2.shape[1],s_encoder2.shape[1])(s_encoder2)
    s_gfup     = layers.Dense(filter_6[0])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[1])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder2 = layers.Add()([s_encoder2,s_gfup])
    s_encoder2 = LeakyReLU()(s_encoder2)
    s_encoder2 = layers.Conv2D(filters = filter_6[1], kernel_size = 4, padding="same")(s_encoder2)
    s_encoder2 = layers.BatchNormalization()(s_encoder2)
    s_encoder2 = layers.Activation("selu")(s_encoder2) 

    s_encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder2)
    s_gfdown   = layers.AveragePooling2D(s_encoder3.shape[1],s_encoder3.shape[1])(s_encoder3)
    s_gfup     = layers.Dense(filter_6[1])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[2])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder3 = layers.Add()([s_encoder3,s_gfup])
    s_encoder3 = LeakyReLU()(s_encoder3)
    s_encoder3 = layers.Conv2D(filters = filter_6[2], kernel_size = 4, padding="same")(s_encoder3)
    s_encoder3 = layers.BatchNormalization()(s_encoder3)
    s_encoder3 = layers.Activation("selu")(s_encoder3) 

    s_encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder3)
    s_gfdown   = layers.AveragePooling2D(s_encoder4.shape[1],s_encoder4.shape[1])(s_encoder4)
    s_gfup     = layers.Dense(filter_6[2])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[3])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder4 = layers.Add()([s_encoder4,s_gfup])
    s_encoder4 = LeakyReLU()(s_encoder4)
    s_encoder4 = layers.Conv2D(filters = filter_6[3], kernel_size = 4, padding="same")(s_encoder4)
    s_encoder4 = layers.BatchNormalization()(s_encoder4)
    s_encoder4 = layers.Activation("selu")(s_encoder4)

    s_encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder4)
    s_gfdown   = layers.AveragePooling2D(s_encoder5.shape[1],s_encoder4.shape[1])(s_encoder5)
    s_gfup     = layers.Dense(filter_6[3])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[4])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder5 = layers.Add()([s_encoder5,s_gfup])
    s_encoder5 = LeakyReLU()(s_encoder5)
    s_encoder5 = layers.Conv2D(filters = filter_6[4], kernel_size = 4, padding="same")(s_encoder5)
    s_encoder5 = layers.BatchNormalization()(s_encoder5)
    s_encoder5 = layers.Activation("selu")(s_encoder5)

    s_encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder5)
    s_gfdown   = layers.AveragePooling2D(s_encoder6.shape[1],s_encoder6.shape[1])(s_encoder6)
    s_gfup     = layers.Dense(filter_6[4])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[5])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder6 = layers.Add()([s_encoder6,s_gfup])
    s_encoder6 = LeakyReLU()(s_encoder6)    
    s_encoder6 = layers.Conv2D(filters = filter_6[5], kernel_size = 4, padding="same")(s_encoder6)
    s_encoder6 = layers.BatchNormalization()(s_encoder6)
    s_encoder6 = layers.Activation("selu")(s_encoder6)

    s_encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder6)
    s_gfdown   = layers.AveragePooling2D(s_encoder7.shape[1],s_encoder7.shape[1])(s_encoder7)
    s_gfup     = layers.Dense(filter_6[5])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[6])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_encoder7 = layers.Add()([s_encoder7,s_gfup])
    s_encoder7 = LeakyReLU()(s_encoder7)    
    s_encoder7 = layers.Conv2D(filters = filter_6[6], kernel_size = 4, padding="same")(s_encoder7)
    s_encoder7 = layers.BatchNormalization()(s_encoder7)
    s_encoder7 = layers.Activation("selu")(s_encoder7)

    s_bottom   = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(s_encoder7)
    s_gfdown   = layers.AveragePooling2D(s_bottom.shape[1],s_bottom.shape[1])(s_bottom)
    s_gfup     = layers.Dense(filter_6[6])(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[7])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_bottom   = layers.Add()([s_bottom,s_gfup])
    s_bottom   = LeakyReLU()(s_bottom)    
    s_bottom   = layers.Conv2D(filters = filter_6[7], kernel_size = 4, padding="same")(s_bottom)
    s_bottom   = layers.BatchNormalization()(s_bottom)
    s_bottom   = layers.Activation("selu")(s_bottom) 
    s_bottom   = layers.Conv2DTranspose(filters = filter_6[8],kernel_size=2,strides = 2, padding= "same")(s_bottom)

    s_decoder7 = layers.Concatenate()([s_encoder7,s_bottom])
    s_gfdown   = layers.AveragePooling2D(s_decoder7.shape[1],s_decoder7.shape[1])(s_decoder7)
    s_gfup     = layers.Dense(filter_6[8]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[8])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder7 = layers.Add()([s_decoder7,s_gfup])
    s_decoder7 = LeakyReLU()(s_decoder7)    
    s_decoder7 = layers.Conv2D(filters = filter_6[8], kernel_size = 4, padding="same")(s_decoder7)
    s_decoder7 = layers.BatchNormalization()(s_decoder7)
    s_decoder7 = layers.Activation("selu")(s_decoder7) 
    s_decoder7 = layers.Conv2DTranspose(filters = filter_6[9],kernel_size=2,strides = 2, padding= "same")(s_decoder7)

    s_decoder6 = layers.Concatenate()([s_encoder6,s_decoder7])
    s_gfdown   = layers.AveragePooling2D(s_decoder6.shape[1],s_decoder6.shape[1])(s_decoder6)
    s_gfup     = layers.Dense(filter_6[9]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[9])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder6 = layers.Add()([s_decoder6,s_gfup])
    s_decoder6 = LeakyReLU()(s_decoder6)    
    s_decoder6 = layers.Conv2D(filters = filter_6[9], kernel_size = 4, padding="same")(s_decoder6)
    s_decoder6 = layers.BatchNormalization()(s_decoder6)
    s_decoder6 = layers.Activation("selu")(s_decoder6) 
    s_decoder6 = layers.Conv2DTranspose(filters = filter_6[10],kernel_size=2,strides = 2, padding= "same")(s_decoder6)

    s_decoder5 = layers.Concatenate()([s_encoder5,s_decoder6])
    s_gfdown   = layers.AveragePooling2D(s_decoder5.shape[1],s_decoder5.shape[1])(s_decoder5)
    s_gfup     = layers.Dense(filter_6[10]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[10])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder5 = layers.Add()([s_decoder5,s_gfup])
    s_decoder5 = LeakyReLU()(s_decoder5) 
    s_decoder5 = layers.Conv2D(filters = filter_6[10], kernel_size = 4, padding="same")(s_decoder5)
    s_decoder5 = layers.BatchNormalization()(s_decoder5)
    s_decoder5 = layers.Activation("selu")(s_decoder5) 
    s_decoder5 = layers.Conv2DTranspose(filters = filter_6[11],kernel_size=2,strides = 2, padding= "same")(s_decoder5)
    
    s_decoder4 = layers.Concatenate()([s_encoder4,s_decoder5])
    s_gfdown   = layers.AveragePooling2D(s_decoder4.shape[1],s_decoder4.shape[1])(s_decoder4)
    s_gfup     = layers.Dense(filter_6[11]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[11])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder4 = layers.Add()([s_decoder4,s_gfup])
    s_decoder4 = LeakyReLU()(s_decoder4) 
    s_decoder4 = layers.Conv2D(filters = filter_6[11], kernel_size = 4, padding="same")(s_decoder4)
    s_decoder4 = layers.BatchNormalization()(s_decoder4)
    s_decoder4 = layers.Activation("selu")(s_decoder4) 
    s_decoder4 = layers.Conv2DTranspose(filters = filter_6[12],kernel_size=2,strides = 2, padding= "same")(s_decoder4)

    s_decoder3 = layers.Concatenate()([s_encoder3,s_decoder4])
    s_gfdown   = layers.AveragePooling2D(s_decoder3.shape[1],s_decoder3.shape[1])(s_decoder3)
    s_gfup     = layers.Dense(filter_6[12]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[12])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder3 = layers.Add()([s_decoder3,s_gfup])
    s_decoder3 = LeakyReLU()(s_decoder3) 
    s_decoder3 = layers.Conv2D(filters = filter_6[12], kernel_size = 4, padding="same")(s_decoder3)
    s_decoder3 = layers.BatchNormalization()(s_decoder3)
    s_decoder3 = layers.Activation("selu")(s_decoder3) 
    s_decoder3 = layers.Conv2DTranspose(filters = filter_6[13],kernel_size=2,strides = 2, padding= "same")(s_decoder3)

    s_decoder2 = layers.Concatenate()([s_encoder2,s_decoder3])
    s_gfdown   = layers.AveragePooling2D(s_decoder2.shape[1],s_decoder2.shape[1])(s_decoder2)
    s_gfup     = layers.Dense(filter_6[13]*2)(s_gf)
    s_gf       = layers.Concatenate()([s_gf,s_gfdown])
    s_gf       = layers.Dense(filter_6[13])(s_gf)
    s_gf       = layers.Activation('selu')(s_gf)
    s_decoder2 = layers.Add()([s_decoder2,s_gfup])
    s_decoder2 = LeakyReLU()(s_decoder2) 
    s_decoder2 = layers.Conv2D(filters = filter_6[13], kernel_size = 4, padding="same")(s_decoder2)
    s_decoder2 = layers.BatchNormalization()(s_decoder2)
    s_decoder2 = layers.Activation("selu")(s_decoder2) 
    s_decoder2 = layers.Conv2DTranspose(filters = filter_6[14],kernel_size=2,strides = 2, padding= "same")(s_decoder2)  

    s_decoder1 = layers.Concatenate()([s_encoder1,s_decoder2])
    s_gfup     = layers.Dense(filter_6[14]*2)(s_gf)
    s_decoder1 = layers.Add()([s_decoder1,s_gfup])
    s_decoder1 = LeakyReLU()(s_decoder1) 
    s_decoder1 = layers.Conv2D(filters = filter_6[14], kernel_size = 4, padding="same")(s_decoder1)
    s_decoder1 = layers.BatchNormalization()(s_decoder1)
    s_decoder1 = layers.Activation("selu")(s_decoder1) 

    s_outputs = layers.Conv2D(6, kernel_size = 1, activation="tanh", padding="same")(s_decoder1)


#================
#==== output ====
#================

    outputs = layers.Concatenate()([a_outputs,s_outputs])
    model = keras.Model(inputs, outputs)
    return model

model = SVBRDF_partial_branched()
model.summary()