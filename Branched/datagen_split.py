import tensorflow as tf
from datetime import datetime
import random
#

NN_size = 256
batch_size = 8

def logrithm(img):
    return tf.math.log(100.*img + 1.)/tf.math.log(101.)  

def imagestack_img(img,mode):
    shape = img.shape[0]
    inputimg = img[:,0:shape,:] #rendered 3 channels

    if (mode == 'a'):
        albedo  = img[:,shape*2:shape*3,: ]
        return tf.concat([logrithm(inputimg), albedo],axis = -1)
    elif (mode == 's'):
        specular= img[:,shape*4:shape*5,: ]
        return tf.concat([logrithm(inputimg), specular],axis = -1)
    elif (mode == 'n'):
        normal  = img[:,shape  :shape*2,:2]
        return tf.concat([logrithm(inputimg), normal],axis = -1)
    elif (mode == 'r'):
        rough   = img[:,shape*3:shape*4,0 ]
        rough = tf.expand_dims(rough,axis=-1)
        return tf.concat([logrithm(inputimg), rough],axis = -1)
    else:
        print('wtf, again?')

def img_process(raw,mode):
    img_stack = imagestack_img(raw,mode)
    ran_seed = random.seed(datetime.now())
    tf.random.set_seed(ran_seed)
    img_stack = tf.image.random_crop(img_stack,  [NN_size, NN_size, img_stack.shape[2]])
    img_stack = img_stack*2-1
    return img_stack[:,:,0:3], img_stack[:,:,3:]

def parse_func(path,mode):
    image_string = tf.io.read_file(path)
    raw_input = tf.image.decode_image(image_string,dtype = tf.float32)
    ins, outs = tf.py_function(func = img_process(mode), inp = [raw_input],Tout =(tf.float32,tf.float32) )
    ins.set_shape((256,256,3))
    outs.set_shape((256,256,9))
    return ins,outs

def svbrdf_gen(path, bs, mode):
    dataset = tf.data.Dataset.list_files(path+'/*.png')
    trainset = dataset.map(parse_func(mode))
    trainset = trainset.repeat()
    trainset = trainset.batch(bs)
    return trainset
















