
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
#
def SVBRDF(num_classes):
    #=============== first layer ==================

    inputs = keras.Input(shape=(256,256) + (3,))

    #GF = layers.LeakyReLU()(inputs)
    GF = layers.AveragePooling2D((inputs.shape[1],inputs.shape[1]))(inputs)
    GF = layers.Dense(128)(GF)
    GF = layers.Activation('selu')(GF)

    x = layers.SeparableConv2D(128,4, 2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    #previous_block_activation = x  # Set aside residual

    #========== define filters for unet ===================

    downfilters = np.array([128,256,512,512,512,512,512])
    Upfilters = np.flip(np.copy(downfilters))
    downfilters = np.delete(downfilters,0)
    prefilter = 128

    #===================== upsampling =======================

    for filters in downfilters:
        #print(x.shape)
        #print(filters)
        GFdown = layers.AveragePooling2D((x.shape[1],x.shape[1]))(x)
        GFup   = layers.Dense(prefilter)(GF)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.SeparableConv2D(filters, 4,2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        prefilter = filters

    #====================== downsampling ============================

    for filters in Upfilters:

        GFdown = layers.AveragePooling2D((x.shape[1],x.shape[1]))(x)
        GFup   = layers.Dense(prefilter)(GF)
        GF     = layers.Concatenate()([GF,GFdown])
        GF = layers.Dense(filters)(GF)
        GF = layers.Activation('selu')(GF)
        
        x = layers.Add()([x,GFup])
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(filters, 4,2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        prefilter = filters

    #====================== last connection =====================

    GFup   = layers.Dense(prefilter)(x)
    x = layers.Add()([x,GFup])
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def UNET_exact (num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(filters = 64, kernel_size = 4, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for fltrs in [64, 128,128,256,512,512,1024]:
        x = layers.Activation("selu")(x)
        x = layers.Conv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("selu")(x)
        x = layers.Conv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(fltrs, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    #[256, 128, 64, 32]
    for fltrs in [1024,512,512,256,128,128,64,64]:
        x = layers.Activation("selu")(x)
        x = layers.Conv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("selu")(x)
        x = layers.Conv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        #print(residual.shape)
        residual = layers.Conv2D(fltrs, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)
    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def UNET_1cnn( num_classes):

    inputs = keras.Input(shape=(256,256) + (3,))

    ### [First half of the network: downsampling inputs] ###

    #[128,256,512,512,512,512,512,512]
    #[512,512,512,512,512,512,256,128]

    # Entry block
    x = layers.Conv2D(filters = 128, kernel_size = 4, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for fltrs in [256,512,512,512,512,512,512]:

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters = fltrs, kernel_size = 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        '''
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters = fltrs, kernel_size = 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        '''
        #x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
      
        # Project residual
        residual = layers.Conv2D(filters = fltrs, kernel_size = 4, strides=2, padding="same")(
            previous_block_activation
        )

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    
    for fltrs in [512,512,512,512,512,512,256,128]:
        print(x.shape)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters = fltrs, kernel_size = 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        print(x.shape)
        #print('1'+str(x.shape))
        #print(previous_block_activation.shape)
        
        #x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        
        residual = layers.Conv2D(filters = fltrs, kernel_size = 4, strides=1, padding="same")(residual)
        print(residual.shape)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    # Add a per-pixel classification layer

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def UNET_paper(num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    encoder1 = layers.Conv2D(filters = 64, kernel_size = 4, padding="same")(inputs)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("relu")(encoder1) 

    encoder1 = layers.Conv2D(filters = 64, kernel_size = 4, padding="same")(encoder1)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("relu")(encoder1)

    encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder1)


    encoder2 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("relu")(encoder2) 

    encoder2 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("relu")(encoder2)

    encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder2)


    encoder3 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("relu")(encoder3) 

    encoder3 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("relu")(encoder3)

    encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder3)

    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("relu")(encoder4) 

    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("relu")(encoder4)


    encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder4)

    encoder5 = layers.Conv2D(filters = 1024, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("relu")(encoder5) 

    encoder5 = layers.Conv2D(filters = 1024, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("relu")(encoder5)

    encoder5 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(encoder5)

    
    decoder4 = layers.Concatenate()([encoder4,encoder5])

    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("relu")(decoder4) 

    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("relu")(decoder4)

    decoder4 = layers.Conv2DTranspose(filters = 256,kernel_size=2,strides = 2, padding= "same")(decoder4)


    decoder3 = layers.Concatenate()([encoder3,decoder4])

    decoder3 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("relu")(decoder3) 

    decoder3 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("relu")(decoder3)

    decoder3 = layers.Conv2DTranspose(filters = 128,kernel_size=2,strides = 2, padding= "same")(decoder3)


    decoder2 = layers.Concatenate()([encoder2,decoder3])

    decoder2 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("relu")(decoder2) 

    decoder2 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("relu")(decoder2)

    decoder2 = layers.Conv2DTranspose(filters = 64,kernel_size=2,strides = 2, padding= "same")(decoder2)  


    decoder1 = layers.Concatenate()([encoder1,decoder2])

    decoder1 = layers.Conv2D(filters = 64, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("relu")(decoder1) 

    decoder1 = layers.Conv2D(filters = 64, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("relu")(decoder1)

    outputs = layers.Conv2D(num_classes, kernel_size = 1, activation="sigmoid", padding="same")(decoder1)

    model = keras.Model(inputs, outputs)
    return model

def UNET_paper2(num_classes):
    inputs = keras.Input(shape=(256,256) + (3,))

    encoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(inputs)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("selu")(encoder1) 
    encoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(encoder1)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation("selu")(encoder1)

    encoder2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder1)
    encoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("selu")(encoder2) 
    encoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2 = layers.Activation("selu")(encoder2)

    encoder3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder2)
    encoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("selu")(encoder3) 
    encoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation("selu")(encoder3)

    encoder4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder3)
    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("selu")(encoder4)
    encoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation("selu")(encoder4)

    encoder5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder4)
    encoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("selu")(encoder5)
    encoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder5)
    encoder5 = layers.BatchNormalization()(encoder5)
    encoder5 = layers.Activation("selu")(encoder5)

    encoder6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder5)
    encoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder6)
    encoder6 = layers.BatchNormalization()(encoder6)
    encoder6 = layers.Activation("selu")(encoder6)
    encoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder6)
    encoder6 = layers.BatchNormalization()(encoder6)
    encoder6 = layers.Activation("selu")(encoder6)

    encoder7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder6)
    encoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder7)
    encoder7 = layers.BatchNormalization()(encoder7)
    encoder7 = layers.Activation("selu")(encoder7)
    encoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(encoder7)
    encoder7 = layers.BatchNormalization()(encoder7)
    encoder7 = layers.Activation("selu")(encoder7)

    bottom = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(encoder7)
    bottom = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(bottom)
    bottom = layers.BatchNormalization()(bottom)
    bottom = layers.Activation("selu")(bottom) 
    bottom = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(bottom)
    bottom = layers.BatchNormalization()(bottom)
    bottom = layers.Activation("selu")(bottom)
    bottom = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(bottom)

    decoder7 = layers.Concatenate()([encoder7,bottom])
    decoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder7)
    decoder7 = layers.BatchNormalization()(decoder7)
    decoder7 = layers.Activation("selu")(decoder7) 
    decoder7 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder7)
    decoder7 = layers.BatchNormalization()(decoder7)
    decoder7 = layers.Activation("selu")(decoder7)
    decoder7 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder7)

    decoder6 = layers.Concatenate()([encoder6,decoder7])
    decoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder6)
    decoder6 = layers.BatchNormalization()(decoder6)
    decoder6 = layers.Activation("selu")(decoder6) 
    decoder6 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder6)
    decoder6 = layers.BatchNormalization()(decoder6)
    decoder6 = layers.Activation("selu")(decoder6)
    decoder6 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder6)

    decoder5 = layers.Concatenate()([encoder5,decoder6])
    decoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder5)
    decoder5 = layers.BatchNormalization()(decoder5)
    decoder5 = layers.Activation("selu")(decoder5) 
    decoder5 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder5)
    decoder5 = layers.BatchNormalization()(decoder5)
    decoder5 = layers.Activation("selu")(decoder5)
    decoder5 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder5)
    
    decoder4 = layers.Concatenate()([encoder4,decoder5])
    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("selu")(decoder4) 
    decoder4 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation("selu")(decoder4)
    decoder4 = layers.Conv2DTranspose(filters = 512,kernel_size=2,strides = 2, padding= "same")(decoder4)

    decoder3 = layers.Concatenate()([encoder3,decoder4])
    decoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("selu")(decoder3) 
    decoder3 = layers.Conv2D(filters = 512, kernel_size = 4, padding="same")(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation("selu")(decoder3)
    decoder3 = layers.Conv2DTranspose(filters = 256,kernel_size=2,strides = 2, padding= "same")(decoder3)

    decoder2 = layers.Concatenate()([encoder2,decoder3])
    decoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("selu")(decoder2) 
    decoder2 = layers.Conv2D(filters = 256, kernel_size = 4, padding="same")(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation("selu")(decoder2)
    decoder2 = layers.Conv2DTranspose(filters = 128,kernel_size=2,strides = 2, padding= "same")(decoder2)  

    decoder1 = layers.Concatenate()([encoder1,decoder2])
    decoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("selu")(decoder1) 
    decoder1 = layers.Conv2D(filters = 128, kernel_size = 4, padding="same")(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation("selu")(decoder1)

    outputs = layers.Conv2D(num_classes, kernel_size = 1, activation="tanh", padding="same")(decoder1)

    model = keras.Model(inputs, outputs)
    return model


