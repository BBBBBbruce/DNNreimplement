
import tensorflow as tf
from svbrdf import SVBRDF, rendering_loss
from DataGen import DataGen


model = SVBRDF(9)
model.summary()

train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'
trainx, trainy = DataGen(train_path)

model.compile(optimizer = 'Adam', loss = rendering_loss, metrics = ['mse'])
model.fit( trainx,trainy,verbose =2 , batch_size=1, epochs=20, validation_split=0.1)
model.save('E:\workspace_ms_zhiyuan\DNNreimplement\Deschaintre\Model_saved')

