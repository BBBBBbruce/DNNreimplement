import tensorflow as tf
from datetime import datetime
import random
#

NN_size = 256
batch_size = 8

def logrithm(img):
    return tf.math.log(100.*img + 1.)/tf.math.log(101.)  

def imagestack_img(img):
    shape = img.shape[0]

    inputimg = img[:,0:shape,:] #rendered 3 channels

    albedo  = img[:,shape*2:shape*3,: ]
    specular= img[:,shape*4:shape*5,: ]
    normal  = img[:,shape  :shape*2,:2]
    rough   = img[:,shape*3:shape*4,0 ]
    rough = tf.expand_dims(rough,axis=-1)

    return logrithm(inputimg), tf.concat([albedo,specular,normal,rough],axis = -1)


def img_process(raw):
    ins,outs = imagestack_img(raw)
    ran_seed = random.seed(datetime.now())
    tf.random.set_seed(ran_seed)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3])
    tf.random.set_seed(ran_seed) 
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])
   
    return inputs, outputs


def parse_func(path):
    image_string = tf.io.read_file(path)
    raw_input = tf.image.decode_image(image_string,dtype = tf.float32)
    ins, outs = tf.py_function(func = img_process, inp = [raw_input],Tout =(tf.float32,tf.float32) )
    ins.set_shape((256,256,3))
    outs.set_shape((256,256,9))
    return ins*2-1,outs*2-1

def svbrdf_gen(path, bs):
    dataset = tf.data.Dataset.list_files(path+'\*.png')
    trainset = dataset.map(parse_func)
    trainset = trainset.repeat()
    #trainset = trainset.skip()
    trainset = trainset.batch(bs)
    return trainset
















