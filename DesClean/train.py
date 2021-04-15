
import tensorflow as tf
from svbrdf import SVBRDF
from DataGen import svbrdf_gen
from GGXrenderer import rendering_loss,l1_loss
from tensorflow.keras.optimizers import Adam 

#tf.keras.backend.floatx()
#os.environ['AUTOGRAPH_VERBOSITY'] = 5
model = SVBRDF(9)
#model.summary()

train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
#test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
test_path = 'D:\Y4\DNNreimplement\Deschaintre\Dataset\Train'
print('load_data')
ds = svbrdf_gen(test_path,8)
print(ds.element_spec)
print('finish_loading')
opt = Adam(lr=0.0002)
model.compile(optimizer = opt, loss = rendering_loss, metrics = ['mse'])
model.fit( ds,verbose =2 , epochs=20) #
model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Model_saved')

