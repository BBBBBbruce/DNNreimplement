import tensorflow as tf
from net import SVBRDF_debugged
from datagen import svbrdf_gen
from loss import rendering_loss
from tensorflow.keras.optimizers import Adam 
import datetime
num_epochs = 20


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="E:\workspace_ms_zhiyuan\\tensorboard_log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1,profile_batch=3)

model = SVBRDF_debugged(9)
learning_rate = 0.00002

sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\Train_smaller'
train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'

print('load_data')
ds = svbrdf_gen(train_path,8)
sample_ds = svbrdf_gen(sample,8)
test_ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')


opt = Adam(lr=learning_rate)
model.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
hitory = model.fit( ds,verbose =1 , steps_per_epoch = 2000, epochs=8,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

oss, acc = model.evaluate(test_ds, verbose=2,steps=10)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_trained\Model_trained\Model_saved_1')