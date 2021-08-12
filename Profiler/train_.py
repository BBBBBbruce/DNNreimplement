import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime

from net import SVBRDF_debugged
from datagen import svbrdf_gen
#from loss import rendering_loss,l1_loss,normalisation,l2_loss
#from loss_fixed_opt import render_loss
from loss_fixed import rendering_loss_linear



log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def display_tbs(svbrdf,epoch):
    svbrdf = (svbrdf+1)/2
    def process(maps):
        return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6], maps[:,:,7:] 

    albedo, specular, rough, normal = process(svbrdf)
    rough = tf.expand_dims(rough,axis=-1)
    padd1 = tf.ones ([256,256,1],dtype = tf.float32)
    N = tf.concat([normal,padd1],axis = -1)
    rough = tf.image.grayscale_to_rgb(rough)
    title = ['albedo', 'specular', 'roughness','normal']
    display_list=[ albedo, specular**(1/2.2), N, rough]
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(display_list, (-1, 256, 256, 3))
        tf.summary.image("svbrdf", images,max_outputs=5,step=epoch)

def show_predictions ( epoch, num=1 ):
    for photo, _ in sample_ds.take(num):

        pred_svbrdf= model.predict(photo)
        #display(photo[0],svbrdf[0])
        #display_tb(photo[0],pred_svbrdf[0])
        #display_tbs(svbrdf[0],epoch)
        display_tbs(pred_svbrdf[0],epoch+1)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions(epoch)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print("what is this model")
model = SVBRDF_debugged(9)

learning_rate = 0.00002
sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\Train_smaller'
train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'

print('load_data')
ds = svbrdf_gen(train_path,8)
sample_ds = svbrdf_gen(sample,8)
test_ds = svbrdf_gen(test_path,8)
print('finish_loading')

opt = Adam(lr=learning_rate)
model.compile(optimizer = opt, loss = rendering_loss_linear, metrics = ['accuracy'])
hitory = model.fit( ds,verbose =1 , steps_per_epoch = 10000, epochs=4,callbacks=[tensorboard_callback,DisplayCallback()]) #24884 DisplayCallback(),tensorboard_callback,DisplayCallback(),

loss, acc = model.evaluate(test_ds, verbose=2,steps=10)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_1')




