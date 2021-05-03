
import tensorflow as tf
from svbrdf import SVBRDF,UNET_exact,UNET_1cnn  
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss,normalisation,l2_loss
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
import numpy as np
num_epochs = 20

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


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def show_predictions ( num=1 ):
    for photo, svbrdf in sample_ds.take(num):

        pred_svbrdf= model.predict(photo)
        display(photo[0],svbrdf[0])
        display(photo[0],pred_svbrdf[0])
        


#tf.keras.backend.floatx()
#os.environ['AUTOGRAPH_VERBOSITY'] = 5
model = UNET_1cnn(9)
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


opt = Adam(lr=0.00002)
model.compile(optimizer = opt, loss = l1_loss, metrics = ['mse'])
hitory = model.fit( ds,verbose =1 , steps_per_epoch = 2000, epochs=20,callbacks=[DisplayCallback()]) #24884

plt.plot(list(range(0, num_epochs)), hitory.history['loss'], label=' Loss',c='r',alpha=0.6)
plt.plot(list(range(0, num_epochs)), hitory.history['mse'], label=' mse',c='b',alpha=0.6)

model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_saved_1')
plt.show()



