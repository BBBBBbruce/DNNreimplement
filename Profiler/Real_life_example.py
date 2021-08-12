import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
import matplotlib.pyplot as plt
from net import SVBRDF_debugged
from datagen import svbrdf_gen
from loss_fixed import rendering_loss_linear



def gamma_inverse(y):
    return (np.exp(y*np.log(101.))-1.)/100.

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
    dataset = tf.data.Dataset.list_files(path+'/mattress.jpg')
    dataset = dataset.shuffle(8, reshuffle_each_iteration=True)
    trainset = dataset.map(parse_func)
    trainset = trainset.repeat()
    trainset = trainset.batch(bs)
    return trainset


def display_predicted(photo,svbrdf):
    photo = (photo+1)/2
    svbrdf = (svbrdf+1)/2
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6], maps[:,:,7:] 

    albedo, specular, rough, normal = process(svbrdf)

    rough = tf.expand_dims(rough,axis=-1)

    padd1 = tf.ones ([256,256,1],dtype = tf.float32)

    N = tf.concat([normal,padd1],axis = -1)

    rough = tf.image.grayscale_to_rgb(rough)

    title = ['photo','albedo', 'specular', 'normal','roughness']
    display_list=[ gamma_inverse(photo), albedo, specular, N, rough]

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + "predicted_real"
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)

def show_predictions (dataset, model, num=1 ):
    for photo in dataset.take(num):
        i = 1
        pred_svbrdf= model.predict(photo)
        display_predicted(photo[i],pred_svbrdf[i])


test_path =  'E:\workspace_ms_zhiyuan\DNNreimplement\Processed'
print('load_data')
ds = photos_loader(test_path,8)
print(ds.element_spec)
print('finish_loading')
opt = Adam(lr=0.00002)
new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_rl', custom_objects = {'rendering_loss' : rendering_loss_linear},compile=False )
new_model.compile(optimizer = opt, loss = rendering_loss_linear, metrics = ['mse'])
show_predictions(ds,new_model,1)



