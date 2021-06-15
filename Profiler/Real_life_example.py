import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import random

from loss_fixed import rendering_loss_linear
#

NN_size = 256
batch_size = 8

def logrithm(img):
    return tf.math.log(100.*img + 1.)/tf.math.log(101.)  

def img_process(raw):
    img_stack = logrithm(raw)
    return img_stack*2-1

def parse_func(path):
    image_string = tf.io.read_file(path)
    raw_input = tf.image.decode_image(image_string,dtype = tf.float32)
    photos = tf.py_function(func = img_process, inp = [raw_input],Tout =(tf.float32) )
    photos.set_shape((256,256,3))
    return photos

def photos_loader(path, bs):
    dataset = tf.data.Dataset.list_files(path+'/*.jpg')
    trainset = dataset.map(parse_func)
    trainset = trainset.repeat()
    trainset = trainset.batch(bs)
    return trainset


def display_predicted(photo,svbrdf):
    photo = (photo+1)/2
    svbrdf = (svbrdf+1)/2
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

    albedo, specular, normal, rough = process(svbrdf)

    rough = tf.expand_dims(rough,axis=-1)

    padd1 = tf.ones ([256,256,1],dtype = tf.float32)

    N = tf.concat([normal,padd1],axis = -1)

    rough = tf.image.grayscale_to_rgb(rough)

    title = ['albedo', 'specular', 'normal','roughness']
    display_list=[ albedo, specular, N, rough]

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + "predicted"
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)

def show_predictions (dataset, model, num=1 ):
    for photo in dataset.take(num):
        pred_svbrdf= model.predict(photo)
        display_predicted(photo[0],pred_svbrdf[0])


test_path =  'D:\Y4\DNNreimplement\Processed'
print('load_data')
ds = photos_loader(test_path,8)
print(ds.element_spec)
print('finish_loading')
opt = Adam(lr=0.00002)
#new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_sigmoid', custom_objects = {'rendering_loss' : rendering_loss_linear},compile=False )
#new_model.compile(optimizer = opt, loss = rendering_loss_linear, metrics = ['mse'])

for photos in ds.take(1):
    for i in range(8):
        plt.imshow(photos[i])
        plt.show()

#show_predictions(ds,new_model,1)



