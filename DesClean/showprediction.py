import tensorflow as tf
from svbrdf import SVBRDF
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss,normalisation
from tensorflow.keras.optimizers import Adam 
import numpy as np
import matplotlib.pyplot as plt

def display(svbrdf):
    
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 
    albedomap, specularmap, normalinmap, roughnessmap = process(svbrdf)
    
    shapex = 256
    shapey = 256

    x = np.linspace(0,shapex-1,shapex)
    y = np.linspace(0,shapey-1,shapey)
    xx,yy = tf.meshgrid(x,y)
    xx = tf.cast(tf.reshape(xx ,(shapex,shapey,1)),dtype = tf.float32)
    yy = tf.cast(tf.reshape(yy ,(shapex,shapey,1)),dtype = tf.float32)
    padd1 = tf.reshape(tf.ones ([shapex,shapey],dtype = tf.float32)*255,(shapex,shapey,1))

    N = normalisation(tf.concat([normalinmap,padd1],axis = -1)/255)

    plt.figure(figsize=(15, 15))
    roughnessmap = tf.expand_dims(roughnessmap,-1)
    #print(albedomap.shape, specularmap.shape, N.shape, roughnessmap.shape)
    title = ['albedo', 'specular', 'normal','roughness']
    display_list=[albedomap, specularmap, N, roughnessmap]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()



def show_predictions (dataset, model, num=1 ):
    for photo, svbrdf in dataset.take(num):
        plt.imshow(tf.keras.preprocessing.image.array_to_img(photo[0]))
        plt.axis('off')
        plt.show()
        pred_svbrdf= model.predict(photo)
        display(svbrdf[0])
        display(pred_svbrdf[0])

print(tf.__version__)

test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
print('load_data')
ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')



opt = Adam(lr=0.00002)
new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_saved', custom_objects = {'rendering_loss' : rendering_loss},compile=False )
#new_model.summary()
new_model.compile(optimizer = opt, loss = rendering_loss, metrics = ['mse'])


show_predictions(ds,new_model,1)




