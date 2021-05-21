
import tensorflow as tf
from svbrdf import SVBRDF, rendering_loss
from DataGen import svbrdf_gen
#from ds_test.ds_test import DsTest
tf.keras.backend.floatx()
model = SVBRDF(9)
#model.summary()

#train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
#test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
#(trainx, trainy),(testx,testy) = DsTest.load()


train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
print('load_data')
ds = svbrdf_gen(train_path,8)
print(ds.element_spec)
print('finish_loading')
model.compile(optimizer = 'Adam', loss = rendering_loss, metrics = ['mse'])
model.fit( ds,verbose =2 , epochs=20)
model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Deschaintre\Model_saved')

