import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import datetime
import sys

from splited_net import net_split
from loss import rendering_loss
from datagen_split import svbrdf_gen


num_epochs = 20

opt = Adam(lr=0.00002)
step_num = 2
epoch_num = 2
#sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\Train_smaller'
train_path = '/vol/bitbucket/zz6117/Data_Deschaintre18/trainBlended'
test_path =  '/vol/bitbucket/zz6117/Data_Deschaintre18/testBlended'
filter_1 = [32,64,128,128,128,128,256,256,256,128,128,128,128,64,32]
filter_2 = [64,128,128,256,256,256,256,256,256,256,256,256,128,128,64]
filter_3 = [64,128,128,128,128,256 ,512,512,512, 256,128,128,128,128,64]


print('load_data')
ds = svbrdf_gen(train_path,8,'s')
#test_ds = svbrdf_gen(test_path,8,'a')
print('finish_loading')

for photo, svbrdf in ds.take(1):
    print(photo.shape)
    print(svbrdf.shape)