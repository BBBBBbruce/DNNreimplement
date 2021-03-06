import numpy as np
import os
import tensorflow as tf
#from imageprocessing import imagestack, imagestack_img
import matplotlib.pyplot as plt
from random import shuffle
import glob

#tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)
NN_size = 256
batch_size = 8

def imagestack_img(img):
    #tf.squeeze(img) # uncomment this in training
    shape = img.shape[0]
    inputimg = img[:,0:shape,:] #rendered 3 channels

    # assemble order:
    #albedo, specular, normal, roughness
    reference = np.zeros([shape,shape,9])
    reference[:,:,0:3] = img[:,shape*2:shape*3,:]
    reference[:,:,3:6] = img[:,shape*4:shape*5,:]
    reference[:,:,6:8] = img[:,shape:shape*2,1:]
    reference[:,:,8]   = img[:,shape*3:shape*4,0]
    #print(img.shape)
    return inputimg, reference

def datagen(path_str):
    #print(path_str)
    image_string = tf.io.read_file(path_str)
    raw_input = tf.image.decode_image(image_string,dtype=tf.float32)
    ins,outs = imagestack_img(raw_input)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3])
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])
    return inputs, outputs

def DataGen(path, bs):
    #load paths
    dataset = tf.data.Dataset.list_files(path+'\*.png')
    '''
    def g(x):
        print(x)
        return 5
    dataset = dataset.map(g)
    # print(dataset.make_one_shot_iterator().next())
    print(next(dataset))
    '''
    #map
    #ds = tf.py_function(func = datagen, inp = [dataset],Tout =(tf.float32,tf.float32) )
    #ds.set_shape(((256,256,3),(256,256,9)))
    ds  = dataset.map(lambda x : tf.py_function(func = datagen, inp = [x],Tout =(tf.float32,tf.float32) ))
    #split the inputs and outputs
    inflow = ds.map(lambda a, b: a)
    outflow = ds.map(lambda a, b: b)
    #inflow = inflow.set_shape(256,256,3)
    #print('inflow shape: '+str(inflow))
    #batch them
    inbatch = inflow.batch(bs)
    outbatch = outflow.batch(bs)
    #print(next(inbatch))
    
    return inbatch, outbatch


def parse_path(path):
    image_string = tf.io.read_file(path)
    raw_input = tf.cast(tf.image.decode_image(image_string),tf.float32)
    return raw_input

def img_process(raw):
    ins,outs = imagestack_img(raw)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3])
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])
    return inputs, outputs

def tf_im_stack_map(raw):
    ins, outs = tf.py_function(func = img_process, inp = [raw],Tout =(tf.float32,tf.float32) )
    ins.set_shape((256,256,3))
    outs.set_shape((256,256,9))
    ins = tf.expand_dims(ins, axis=0)
    outs = tf.expand_dims(outs, axis=0)
    return ins,outs

def svbrdf_gen(path, bs):
    #path = 'E:\workspace_ms_zhiyuan\DNNreimplement\Deschaintre\Dataset\Train'
    dataset = tf.data.Dataset.list_files(path+'\*.png')
    image_ds = dataset.map(parse_path)
    trainset = image_ds.map(tf_im_stack_map)
    trainset.batch(bs)
    return trainset
'''
trainset = svbrdf_gen(path,8)
for ins,outs in trainset.take(2):
    print(ins.shape)
    print(outs.shape)
#ins, outs = DataGen(path,batch_size)

#print(ins,outs)
'''
'''
pathList = glob.glob(os.path.join(path +'\*.png'))
shuffle(pathList)
filenamesTensor = tf.constant(pathList) 
dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)
dataset = dataset.map(datagen, num_parallel_calls=1)
'''














