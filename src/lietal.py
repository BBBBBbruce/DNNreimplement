
from include import *

def encoder(inputsize, num_class):
    #aussume now is 128*128*3?
    inputs = keras.Input(shape=inputsize + (num_class,))
    model = Conv2D(32,kernel_size=6, strides=2, padding= 'same', use_bias = False)(inputs)
    model = BatchNormalization()(model)
    x1 = Activation('relu')(model)
    model = Conv2D(64,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x1)
    model = BatchNormalization()(model)
    x2 = Activation('relu')(model)
    model = Conv2D(128,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x2)
    model = BatchNormalization()(model)
    x3 = Activation('relu')(model)
    model = Conv2D(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x3)
    model = BatchNormalization()(model)
    x4 = Activation('relu')(model)
    model = Conv2D(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x4)
    model = BatchNormalization()(model)
    x5 = Activation('relu')(model)
    model = Conv2D(512,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x5)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    #model = keras.Model(inputs, model)
    #decoder
    return inputs, model, x1, x2, x3,x4,x5

def decoder(inputs, outputs, x1 ,x2, x3, x4, x5):

    model = Conv2DTranspose(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(outputs)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x5])

    model = Conv2DTranspose(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x4])

    model = Conv2DTranspose(128,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x3])

    model = Conv2DTranspose(64,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x2])

    model = Conv2DTranspose(32,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x1])

    model = Conv2DTranspose(64,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(3,kernel_size=5, strides=1, padding= 'same', use_bias = True)(model)
    model = Activation('tanh')(model)

    return keras.Model(inputs, model)

def predict(model, image, id):
    prediction = model.predict(image)
    if(id == 'normal'):
        norm = tf.sqrt(tf.expand_dims(tf.reduce_sum(prediction*prediction, 1),1))
        prediction = prediction/ norm
    elif(id == 'rough'):
        prediction = tf.expand_dims(tf.reduce_mean(prediction,1),1)
    elif(id == 'depth'):
        prediction = tf.expand_dims(tf.reduce_mean(prediction,1),1)
        prediction = 1 / (0.4 * (prediction + 1) + 0.25) 
    return prediction

def envmap(inputsize,num_class,numcoef):
    inputs = keras.Input(shape=inputsize + (num_class,))
    model = Conv2D(1024,kernel_size=(4,4), strides=1, padding= 'valid', use_bias = False)(inputs)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Reshape((model.shape[3],1))(model)
    model = tf.reduce_mean(model,2)
    model = Sequential([
        Dense(1024),
        Activation('relu'),
        Dropout(0.25),
        Dense(numcoef*3)
    ])(model)
    model = tf.math.tanh(model)
    model = Reshape((3,numcoef))(model)
    return keras.Model(inputs, model)

def globalillum(inputsize,num_class):
    inputs = keras.Input(shape=inputsize + (num_class,))
    model = Conv2D(32,kernel_size=6, strides=2, padding= 'same', use_bias = False)(inputs)
    model = BatchNormalization()(model)
    x1 = Activation('relu')(model)
    #model = Conv2D(64,kernel_size=16,  strides=2, padding= 'same', use_bias = False)(x1)
    model = Conv2D(64,kernel_size=6, dilation_rate = 3, strides=1, padding= 'same', use_bias = False)(x1)
    model = BatchNormalization()(model)
    x2 = Activation('relu')(model)
    #model = Conv2D(128,kernel_size=16,  strides=2, padding= 'same', use_bias = False)(x2)
    model = Conv2D(128,kernel_size=6, dilation_rate = 3, strides=1, padding= 'same', use_bias = False)(x2)
    model = BatchNormalization()(model)
    x3 = Activation('relu')(model)
    model = Conv2D(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x3)
    model = BatchNormalization()(model)
    x4 = Activation('relu')(model)
    model = Conv2D(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(x4)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(256,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x4])
    model = Conv2DTranspose(128,kernel_size=4, strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x3])
    #model = Conv2DTranspose(64,kernel_size=16, strides=2, padding= 'same', use_bias = False)(model)
    model = Conv2DTranspose(64,kernel_size=6, dilation_rate = 3, strides=1, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x2])
    #model = Conv2DTranspose(32,kernel_size=16,  strides=2, padding= 'same', use_bias = False)(model)
    model = Conv2DTranspose(32,kernel_size=6, dilation_rate = 3, strides=1, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Concatenate()([model,x1])
    model = Conv2DTranspose(64,kernel_size=4,  strides=2, padding= 'same', use_bias = False)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)


    model = Conv2D(3,kernel_size=5, strides=1, padding= 'same', use_bias = False)(model)
    model = Activation('tanh')(model)
    return keras.Model(inputs, model)

def residual(inputsize,num_class):
    inputs = keras.Input(shape=inputsize + (num_class,))
    model =  Conv2D(128,kernel_size=3, dilation_rate = 2,strides=1, padding= 'same', use_bias = False)(inputs)
    
    return keras.Model(inputs, model)

inputs, outputs, x1 ,x2, x3, x4, x5 = encoder((256,256),4)
test1 = decoder(inputs, outputs, x1 ,x2, x3, x4, x5)
test2 = envmap((4,4),512, 9)
test3 = globalillum((256,256),12)

test3.summary()


