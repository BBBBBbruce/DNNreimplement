from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


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

class EntryBlock(keras.layers.Layer):
    def __init__(self,fltr):
        super(EntryBlock, self).__init__()
        self.fltr = fltr
        self.conv1 = Conv2d(self.fltr)
        self.conv2 = Conv2d(self.fltr)
        
    def call(self, inputs):
        encoder = self.conv1(inputs)
        encoder = self.conv2(encoder)
        return encoder

class EncoderBlock(keras.layers.Layer):
    def __init__(self,fltr):
        super(EncoderBlock, self).__init__()
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

class BottomBlock(keras.layers.Layer):
    def __init__(self,fltr):
        super(BottomBlock, self).__init__()
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

class DecoderBlock(keras.layers.Layer):
    def __init__(self,fltr,next_fltr):
        super(DecoderBlock, self).__init__()
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

class ExitBlock(keras.layers.Layer):
    def __init__(self,fltr):
        super(ExitBlock, self).__init__()
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

class SingleBranch(keras.layers.Layer):
    def __init__(self):
        super(SingleBranch, self).__init__()
        self.encode1 = EntryBlock(128)
        self.encode2 = EncoderBlock(256)
        self.encode3 = EncoderBlock(512)
        self.encode4 = EncoderBlock(512)
        self.encode5 = EncoderBlock(512)
        self.encode6 = EncoderBlock(512)
        self.encode7 = EncoderBlock(512)
        self.bottom  = BottomBlock(512)
        self.decode7 = DecoderBlock(512,512)
        self.decode6 = DecoderBlock(512,512)
        self.decode5 = DecoderBlock(512,512)
        self.decode4 = DecoderBlock(512,512)
        self.decode3 = DecoderBlock(512,256)
        self.decode2 = DecoderBlock(256,128)
        self.decode1 = ExitBlock(128)
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




