#log_dir = "E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
import tensorflow as tf

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

