import tensorflow as tf 
import numpy as np

x = tf.ones([256,256,1])
#x = tf.expand_dims(x)
print(x.shape)