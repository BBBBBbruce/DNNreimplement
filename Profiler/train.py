import tensorflow as tf
from net import SVBRDF_debugged
from datagen import svbrdf_gen
from loss import rendering_loss
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
num_epochs = 20

#log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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
    display_list=[ albedo, specular**(1/2.2), N, rough]

    
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=epoch)

def show_predictions ( epoch, num=1 ):
    for photo, svbrdf in sample_ds.take(num):

        pred_svbrdf= model.predict(photo)
        display_tbs(pred_svbrdf[0],epoch+1)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions(epoch)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="D:\Y4\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1,profile_batch=3)

model = SVBRDF_debugged(9)
learning_rate = 0.00002

#sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\Train_smaller'
train_path = 'D:\Y4\DNNreimplement\Deschaintre\Dataset\Train_smaller\Train_smaller'
#test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'

print('load_data')
ds = svbrdf_gen(train_path,8)
#sample_ds = svbrdf_gen(sample,8)
#test_ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')


opt = Adam(lr=learning_rate)
model.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
hitory = model.fit( ds,verbose =1 , steps_per_epoch = 10, epochs=2,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

#oss, acc = model.evaluate(test_ds, verbose=2,steps=10)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_1')