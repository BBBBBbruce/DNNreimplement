from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

# Unet version doulbe convs, svbrdf version single convs

class GN_Mean(keras.layers.Layer):
    def __init__(self):
        super(GN_Mean, self).__init__()

    def call(self, inputs):
        mean,_ = tf.nn.moments(inputs, axes=[1, 2], keepdims=False)
        return mean

class GN_Mean2(keras.layers.Layer):
    def __init__(self):
        super(GN_Mean2, self).__init__()

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        return mean

class Conv2d(keras.layers.Layer):
    def __init__(self,fltr):
        super(Conv2d, self).__init__()
        self.fltr = fltr
        self.conv = layers.Conv2D(filters = self.fltr, kernel_size = 4, padding="same")
        self.bn = layers.BatchNormalization()
        self.selu = layers.Activation("selu")

    def call(self, inputs):
        convolution = self.conv(inputs)
        convolution = self.bn(convolution)
        convolution = self.selu(convolution)
        return convolution

class EntryBlock_u(keras.layers.Layer):
    def __init__(self,fltr):
        super(EntryBlock_u, self).__init__()
        self.fltr = fltr
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
        
    def call(self, inputs):
        encoder = self.conv1(inputs)
        encoder = self.conv2(encoder)
        return encoder

class EncoderBlock_u(keras.layers.Layer):
    def __init__(self,fltr):
        super(EncoderBlock_u, self).__init__()
        self.fltr = fltr
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
    def call(self, inputs):
        #maxpool layer
        encoder = self.maxpool(inputs)
        encoder = self.conv1(encoder)
        encoder = self.conv2(encoder)
        return encoder

class BottomBlock_u(keras.layers.Layer):
    def __init__(self,fltr):
        super(BottomBlock_u, self).__init__()
        self.fltr = fltr
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
        self.convT = layers.Conv2DTranspose(filters = self.fltr,kernel_size=2,strides = 2, padding= "same")
    def call(self, inputs):
        # fltr = 512
        bottom = self.maxpool(inputs)
        bottom = self.conv1(bottom)
        bottom = self.conv2(bottom)
        bottom = self.convT(bottom)
        return bottom

class DecoderBlock_u(keras.layers.Layer):
    def __init__(self,fltr,next_fltr):
        super(DecoderBlock_u, self).__init__()
        self.fltr = fltr
        self.next_fltr = next_fltr
        self.concat = layers.Concatenate()
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
        self.convT = layers.Conv2DTranspose(filters = self.next_fltr,kernel_size=2,strides = 2, padding= "same")
    def call(self, inputs):
        # fltr = 512
        decoder = self.concat(inputs)
        decoder = self.conv1(decoder)
        decoder = self.conv2(decoder)
        decoder = self.convT(decoder)
        return decoder

class ExitBlock_u(keras.layers.Layer):
    def __init__(self,fltr):
        super(ExitBlock_u, self).__init__()
        self.fltr = fltr
        self.concat = layers.Concatenate()
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
    def call(self, inputs):
        # fltr = 512
        decoder = self.concat(inputs)
        decoder = self.conv1(decoder)
        decoder = self.conv2(decoder)
        return decoder

class SingleBranch_u(keras.layers.Layer):
    def __init__(self,filters):
        super(SingleBranch_u, self).__init__()
        self.encode1 = EntryBlock_u(filters[0])
        self.encode2 = EncoderBlock_u(filters[1])
        self.encode3 = EncoderBlock_u(filters[2])
        self.encode4 = EncoderBlock_u(filters[3])
        self.encode5 = EncoderBlock_u(filters[4])
        self.encode6 = EncoderBlock_u(filters[5])
        self.encode7 = EncoderBlock_u(filters[6])
        self.bottom  = BottomBlock_u(filters[7])
        self.decode7 = DecoderBlock_u(filters[8],filters[9])
        self.decode6 = DecoderBlock_u(filters[9],filters[10])
        self.decode5 = DecoderBlock_u(filters[10],filters[11])
        self.decode4 = DecoderBlock_u(filters[11],filters[12])
        self.decode3 = DecoderBlock_u(filters[12],filters[13])
        self.decode2 = DecoderBlock_u(filters[13],filters[14])
        self.decode1 = ExitBlock_u(filters[14])
    def call(self, inputs):
        encoder1 = self.encode1(inputs)
        encoder2 = self.encode2(encoder1)
        encoder3 = self.encode3(encoder2)
        encoder4 = self.encode4(encoder3)
        encoder5 = self.encode5(encoder4)
        encoder6 = self.encode6(encoder5)
        encoder7 = self.encode7(encoder6)
        bottom   = self.bottom(encoder7)
        decoder7 = self.decode7([encoder7,bottom])
        decoder6 = self.decode6([encoder6,decoder7])
        decoder5 = self.decode5([encoder5,decoder6])
        decoder4 = self.decode4([encoder4,decoder5])
        decoder3 = self.decode3([encoder3,decoder4])
        decoder2 = self.decode2([encoder2,decoder3])
        decoder1 = self.decode1([encoder1,decoder2])
        return decoder1

