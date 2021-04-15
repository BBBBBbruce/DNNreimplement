import numpy as np
import os
import time
from numpy.lib.function_base import append
import tensorflow as tf
from imageprocessing import imagestack, imagestack_img
import matplotlib.pyplot as plt
import cv2 as cv

NN_size = 256
batch_size = 128

print(os.getcwd())
path = os.getcwd()+'\Deschaintre\Dataset\inputExamples\example.png'
img = np.array(cv.imread(path))
shape = img.shape[0]
inputimg = img[:,0:shape,:] #rendered 3 channels

albedo= img[:,shape*2:shape*3,:]
specular = img[:,shape*4:shape*5,:]
normal = img[:,shape:shape*2,1:]
roughness  = tf.reshape(img[:,shape*3:shape*4,0],(shape,shape,1))

svbrdf = tf.concat([albedo,specular,normal,roughness],axis = -1)
print(svbrdf.shape)
'''
tf.compat.v1.enable_eager_execution()
print(os.getcwd())
path = os.getcwd()+'\Deschaintre\Dataset\Train'

dataset = tf.data.Dataset.list_files(path+'/*.png')

def datagen(path_str):
    image_string = tf.io.read_file(path_str)
    raw_input = tf.image.decode_image(image_string,dtype=tf.float64)
    ins,outs = imagestack_img(raw_input)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3])
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])
    return inputs, outputs

dataset  = dataset.map(lambda x : tf.py_function(func = datagen, inp = [x],Tout =(tf.float64,tf.float64) ))
#dataset = dataset.map(datagen)
inflow = dataset.map(lambda a, b: a)
outflow = dataset.map(lambda a, b: b)
inbatch = inflow.batch(batch_size)
outbatch = outflow.batch(batch_size)
print(len(list(inbatch)))

#print(dataset.element_spec)

'''

'''
for element in dataset:
    image_string = tf.io.read_file(element)
    raw_input = tf.image.decode_image(image_string,dtype=tf.float64)
    ins,outs = imagestack_img(raw_input)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3])
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])    
    plt.imshow(inputs)
    plt.show()
'''
#print(dataset.element_spec)













