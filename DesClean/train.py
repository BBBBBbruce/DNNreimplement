
import tensorflow as tf
from svbrdf import SVBRDF,UNET_exact,UNET_1cnn  ,UNET_paper,UNET_paper2
from svbrdf_reimplement import SVBRDF_debugged
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss,normalisation,l2_loss
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
import numpy as np
import datetime
num_epochs = 20

log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

def display_tbs(svbrdf,epoch):
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

    
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=epoch)

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
    display_list=[photo, albedo, specular, N, rough]

    log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=0)

def show_predictions ( epoch, num=1 ):
    for photo, svbrdf in sample_ds.take(num):

        pred_svbrdf= model.predict(photo)
        #display(photo[0],svbrdf[0])
        #display_tb(photo[0],pred_svbrdf[0])
        #display_tbs(svbrdf[0],epoch)
        display_tbs(pred_svbrdf[0],epoch+1)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions(epoch)

    #print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

  

#file_writer = tf.summary.create_file_writer(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

#tf.keras.backend.floatx()

#os.environ['AUTOGRAPH_VERBOSITY'] = 5
model = SVBRDF_debugged(9)
learning_rate = 0.00002
#model = UNET(9)
#model.summary()

sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\Train_smaller'
train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
#test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
#test_path = 'D:\Y4\DNNreimplement\Deschaintre\Dataset\Train'
print('load_data')
ds = svbrdf_gen(train_path,8)
sample_ds = svbrdf_gen(sample,8)
print(ds.element_spec)
print('finish_loading')

for photo, svbrdf in sample_ds.take(1):

        display_tbs(svbrdf[0],0)

opt = Adam(lr=learning_rate)
model.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
hitory = model.fit( ds,verbose =1 , steps_per_epoch = 2000, epochs=20,callbacks=[tensorboard_callback,DisplayCallback()]) #24884 DisplayCallback()

'''
for photo, svbrdf in sample.take(num):

    pred_svbrdf= model.predict(photo)
    display(photo[0],pred_svbrdf[0])
'''
#plt.plot(list(range(0, num_epochs)), hitory.history['loss'], label=' Loss',c='r',alpha=0.6)
#plt.plot(list(range(0, num_epochs)), hitory.history['mse'], label=' mse',c='b',alpha=0.6)

model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_1')
#plt.show()



