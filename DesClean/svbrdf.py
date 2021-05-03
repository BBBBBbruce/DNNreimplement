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
    x = layers.Conv2D(filters = 128, kernel_size = 4, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for fltrs in [256,512,512,512,512,512,512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(fltrs, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    #[256, 128, 64, 32]
    for fltrs in [512,512,512,512,512,512,256,128]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters = fltrs, kernel_size = 4, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        #print(residual.shape)
        residual = layers.Conv2D(fltrs, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

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

#model = UNET(9)
#model.summary()
