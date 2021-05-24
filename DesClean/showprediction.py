import tensorflow as tf
from svbrdf import SVBRDF
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss,normalisation
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
import matplotlib.pyplot as plt

log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log"

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

def display_tb(photo,svbrdf):
    photo = (photo+1)/2
    svbrdf = (svbrdf+1)/2
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

    albedo, specular, normal, rough = process(svbrdf)

    rough = tf.expand_dims(rough,axis=-1)

    padd1 = tf.ones ([256,256,1],dtype = tf.float32)

    N = tf.concat([normal,padd1],axis = -1)

    rough = tf.image.grayscale_to_rgb(rough)

    title = ['photo','albedo', 'specular', 'normal','roughness']
    display_list=[photo, albedo, specular**(1/1.8), N, rough]

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + "prediction"
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)



def show_predictions (dataset, model, num=1 ):
    for photo, svbrdf in dataset.take(num):
        pred_svbrdf= model.predict(photo)
        #display_tb(photo[0],svbrdf[0])
        display_tb(photo[0],pred_svbrdf[0])


print(tf.__version__)



test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
print('load_data')
ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')

opt = Adam(lr=0.00002)
new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_fully_11200', custom_objects = {'rendering_loss' : rendering_loss},compile=False )
#new_model.summary()
new_model.compile(optimizer = opt, loss = rendering_loss, metrics = ['mse'])



show_predictions(ds,new_model,10)




