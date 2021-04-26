import tensorflow as tf
from svbrdf import SVBRDF
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss
from tensorflow.keras.optimizers import Adam 

print(tf.__version__)

test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
print('load_data')
ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')


opt = Adam(lr=0.00002)
new_model = tf.keras.models.load_model('E:\workspace_ms_zhiyuan\DNNreimplement\Model_saved', custom_objects = {'rendering_loss' : rendering_loss},compile=False )
new_model.summary()
new_model.compile(optimizer = opt, loss = rendering_loss, metrics = ['mse'])
loss, acc = new_model.evaluate(ds, verbose=2,steps=10)

print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
