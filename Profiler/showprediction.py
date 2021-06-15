import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
import matplotlib.pyplot as plt

from net import SVBRDF_debugged
from datagen import svbrdf_gen
from loss_fixed import rendering_loss_linear

log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log"

def gamma_inverse(y):
    return (np.exp(y*np.log(101.))-1.)/100.

def display(photo, svbrdf):
    
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

    albedo, specular, normal, rough = process(svbrdf)

    rough = tf.expand_dims(rough,axis=-1)

    padd1 = tf.ones ([256,256,1],dtype = tf.float32)

    N = tf.concat([normal,padd1],axis = -1)

    title = ['photo','albedo', 'specular', 'normal','roughness']
    display_list=[photo, albedo, specular, N, rough]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        #splt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

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

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + "predicted_pro"
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)

def display_gt(photo,svbrdf):
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
    display_list=[ albedo, specular**(1/1.8), N, rough]

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + "gt"
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)

def show_predictions (dataset, model, num=1 ):
    for photo, svbrdf in dataset.take(num):
        pred_svbrdf= model.predict(photo)
        display_gt(photo[0],svbrdf[0])
        display_predicted(photo[0],pred_svbrdf[0])


print(tf.__version__)

test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\single'
print('load_data')
ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')

opt = Adam(lr=0.00002)

new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_rl', custom_objects = {'rendering_loss' : rendering_loss_linear},compile=False )
new_model.compile(optimizer = opt, loss = rendering_loss_linear, metrics = ['mse'])
show_predictions(ds,new_model,1)



