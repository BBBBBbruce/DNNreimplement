from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

from blocks import Conv2d,GN_Mean2


class EntryBlock_s_conv(keras.layers.Layer):
    def __init__(self,fltr):
        super(EntryBlock_s_conv, self).__init__()
        self.fltr = fltr
        self.conv = Conv2d(self.fltr)

    def get_config(self):
        cfg = super().get_config()
        return cfg   

    def call(self, inputs):
        encoder = self.conv(inputs)
        return encoder

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
class EntryBlock_s_gf(keras.layers.Layer):
    def __init__(self,fltr):
        super(EntryBlock_s_gf, self).__init__()
        self.fltr = fltr
        self.mean = GN_Mean2()
        self.fc = layers.Dense(self.fltr)
        self.selu = layers.Activation('selu')


    def call(self, inputs):
        gf = self.mean(inputs)
        gf = self.fc(gf)
        gf = self.selu(gf)
        return gf

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncoderBlock_s_conv(keras.layers.Layer):
    def __init__(self,fltr,fltr_pre):
        super(EncoderBlock_s_conv, self).__init__()
        self.fltr = fltr
        self.fltr_pre = fltr_pre
        self.fc = layers.Dense(self.fltr_pre)
        self.add = layers.Add()
        self.lrelu = LeakyReLU()
        self.conv = Conv2d(self.fltr)

    def call(self, encoder, gf):
        gf = self.fc(gf)
        encoder = self.add([encoder,gf])
        encoder = self.lrelu(encoder)
        encoder = self.conv(encoder)
        return encoder

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EncoderBlock_s_gf(keras.layers.Layer):
    def __init__(self,fltr):
        super(EncoderBlock_s_gf, self).__init__()
        self.fltr = fltr
        self.mean = GN_Mean2()
        self.concat = layers.Concatenate()
        self.fc = layers.Dense(self.fltr)
        self.selu = layers.Activation('selu')

    def call(self, encoder_current, gf):
        mean = self.mean(encoder_current)
        gf = self.concat([gf,mean])
        gf = self.fc(gf)
        gf = self.selu(gf)
        return gf

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BottomBlock_s_conv(keras.layers.Layer):
    def __init__(self,fltr,fltr_pre):
        super(BottomBlock_s_conv, self).__init__()
        self.fltr = fltr
        self.fltr_pre = fltr_pre
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.fc = layers.Dense(self.fltr_pre)
        self.add = layers.Add()
        self.lrelu = LeakyReLU()
        self.conv = Conv2d(self.fltr)
        self.convT = layers.Conv2DTranspose(filters = self.fltr,kernel_size=2,strides = 2, padding= "same")

    def call(self, bottom, gf):
        gf = self.fc(gf)
        bottom = self.add([bottom,gf])
        bottom = self.lrelu(bottom)
        bottom = self.conv(bottom)
        bottom = self.convT(bottom)

        return bottom
    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BottomBlock_s_gf(keras.layers.Layer):
    def __init__(self,fltr):
        super(BottomBlock_s_gf, self).__init__()
        self.fltr = fltr
        self.mean = GN_Mean2()
        self.concat = layers.Concatenate()
        self.fc = layers.Dense(self.fltr)
        self.selu = layers.Activation('selu')

    def call(self, encoder_current, gf):
        mean = self.mean(encoder_current)
        gf = self.concat([gf,mean])
        gf = self.fc(gf)
        gf = self.selu(gf)
        return gf

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DecoderBlock_s_conv(keras.layers.Layer):
    def __init__(self,fltr,fltr_next):
        super(DecoderBlock_s_conv, self).__init__()
        self.fltr = fltr
        self.fltr_next = fltr_next
        self.fc = layers.Dense(self.fltr*2)
        self.add = layers.Add()
        self.lrelu = LeakyReLU()
        self.conv = Conv2d(self.fltr)
        self.convT = layers.Conv2DTranspose(filters = self.fltr_next,kernel_size=2,strides = 2, padding= "same")

    def call(self, decoder, gf):
        gf = self.fc(gf)
        decoder = self.add([decoder,gf])
        decoder = self.lrelu(decoder)
        decoder = self.conv(decoder)
        decoder = self.convT(decoder)
        return decoder

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DecoderBlock_s_gf(keras.layers.Layer):
    def __init__(self,fltr):
        super(DecoderBlock_s_gf, self).__init__()
        self.fltr = fltr
        self.mean = GN_Mean2()
        self.concat = layers.Concatenate()
        self.fc = layers.Dense(self.fltr)
        self.selu = layers.Activation('selu')

    def call(self, encoder_current, gf):
        mean = self.mean(encoder_current)
        gf = self.concat([gf,mean])
        gf = self.fc(gf)
        gf = self.selu(gf)
        return gf

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ExitBlock_s_conv(keras.layers.Layer):
    def __init__(self,fltr):
        super(ExitBlock_s_conv, self).__init__()
        self.fltr = fltr
        self.fc = layers.Dense(self.fltr*2)
        self.add = layers.Add()
        self.lrelu = LeakyReLU()
        self.conv = Conv2d(self.fltr)
        self.convT = layers.Conv2DTranspose(filters = self.fltr,kernel_size=2,strides = 2, padding= "same")

    def call(self, decoder, gf):
        gf = self.fc(gf)
        decoder = self.add([decoder,gf])
        decoder = self.lrelu(decoder)
        decoder = self.conv(decoder)
        return decoder
    
    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SingleBranch_s(keras.layers.Layer):
    def __init__(self,filters):
        super(SingleBranch_s, self).__init__()
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool5 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool6 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.maxpool7 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.concat1 = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        self.concat3 = layers.Concatenate()
        self.concat4 = layers.Concatenate()
        self.concat5 = layers.Concatenate()
        self.concat6 = layers.Concatenate()
        self.concat7 = layers.Concatenate()
        self.encode1 = EntryBlock_s_conv(filters[0])
        self.gf1     = EntryBlock_s_gf(filters[0]) 
        self.encode2 = EncoderBlock_s_conv(filters[1],filters[0])
        self.gf2     = EncoderBlock_s_gf(filters[1])
        self.encode3 = EncoderBlock_s_conv(filters[2],filters[1])
        self.gf3     = EncoderBlock_s_gf(filters[2])
        self.encode4 = EncoderBlock_s_conv(filters[3],filters[2])
        self.gf4     = EncoderBlock_s_gf(filters[3])
        self.encode5 = EncoderBlock_s_conv(filters[4],filters[3])
        self.gf5     = EncoderBlock_s_gf(filters[4])
        self.encode6 = EncoderBlock_s_conv(filters[5],filters[4])
        self.gf6     = EncoderBlock_s_gf(filters[5])
        self.encode7 = EncoderBlock_s_conv(filters[6],filters[5])
        self.gf7     = EncoderBlock_s_gf(filters[6])
        self.bottom  = BottomBlock_s_conv(filters[7],filters[6])
        self.gf_btm  = BottomBlock_s_gf(filters[7])
        self.decode7 = DecoderBlock_s_conv(filters[8],filters[9])
        self.gf_7    = DecoderBlock_s_gf(filters[8])
        self.decode6 = DecoderBlock_s_conv(filters[9],filters[10])
        self.gf_6    = DecoderBlock_s_gf(filters[9])
        self.decode5 = DecoderBlock_s_conv(filters[10],filters[11])
        self.gf_5    = DecoderBlock_s_gf(filters[10])
        self.decode4 = DecoderBlock_s_conv(filters[11],filters[12])
        self.gf_4    = DecoderBlock_s_gf(filters[11])
        self.decode3 = DecoderBlock_s_conv(filters[12],filters[13])
        self.gf_3    = DecoderBlock_s_gf(filters[12])
        self.decode2 = DecoderBlock_s_conv(filters[13],filters[14])
        self.gf_2    = DecoderBlock_s_gf(filters[13])
        self.decode1 = ExitBlock_s_conv(filters[14])

    def call(self, inputs):

        encoder1 = self.encode1(inputs)
        gf = self.gf1(inputs)

        encoder2_entry = self.maxpool1(encoder1)
        encoder2 = self.encode2(encoder2_entry,gf)
        gf =self.gf2(encoder2_entry,gf)

        encoder3_entry = self.maxpool2(encoder2)
        encoder3 = self.encode3(encoder3_entry,gf)
        gf =self.gf3(encoder3_entry,gf)

        encoder4_entry = self.maxpool3(encoder3)
        encoder4 = self.encode4(encoder4_entry,gf)
        gf =self.gf4(encoder4_entry,gf)

        encoder5_entry = self.maxpool4(encoder4)
        encoder5 = self.encode5(encoder5_entry,gf)
        gf =self.gf5(encoder5_entry,gf)

        encoder6_entry = self.maxpool5(encoder5)
        encoder6 = self.encode6(encoder6_entry,gf)
        gf =self.gf6(encoder6_entry,gf)

        encoder7_entry = self.maxpool6(encoder6)
        encoder7 = self.encode7(encoder7_entry,gf)
        gf =self.gf7(encoder7_entry,gf)

        bottom_entry = self.maxpool7(encoder7)
        bottom   = self.bottom(bottom_entry,gf)
        gf =self.gf_btm(bottom_entry,gf)

        decoder7_entry = self.concat1([encoder7,bottom])
        decoder7 = self.decode7(decoder7_entry,gf)
        gf =self.gf_7(decoder7_entry,gf)

        decoder6_entry = self.concat2([encoder6,decoder7])
        decoder6 = self.decode6(decoder6_entry,gf)
        gf =self.gf_6(decoder6_entry,gf)

        decoder5_entry = self.concat3([encoder5,decoder6])
        decoder5 = self.decode5(decoder5_entry,gf)
        gf =self.gf_5(decoder5_entry,gf)

        decoder4_entry = self.concat4([encoder4,decoder5])
        decoder4 = self.decode4(decoder4_entry,gf)
        gf =self.gf_4(decoder4_entry,gf)

        decoder3_entry = self.concat5([encoder3,decoder4])
        decoder3 = self.decode3(decoder3_entry,gf)
        gf =self.gf_3(decoder3_entry,gf)

        decoder2_entry = self.concat6([encoder2,decoder3])
        decoder2 = self.decode2(decoder2_entry,gf)
        gf =self.gf_2(decoder2_entry,gf)

        decoder1_entry = self.concat7([encoder1,decoder2])
        decoder1 = self.decode1(decoder1_entry,gf)

        return decoder1
  
    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)