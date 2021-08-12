import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
import numpy as np
import datetime
import matplotlib.pyplot as plt

import time
from net import SVBRDF_debugged
from datagen import svbrdf_gen
#from loss import rendering_loss,l1_loss,normalisation,l2_loss
#from loss_fixed_opt import render_loss
from loss_fixed import rendering_loss_linear,GGXtf

sample = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\single'
train_path = 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\trainBlended'
test_path =  'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\testBlended'

print('load_data')
ds = svbrdf_gen(train_path,8)
sample_ds = svbrdf_gen(sample,8)
test_ds = svbrdf_gen(test_path,8)
print('finish_loading')

lightpos_set = np.array([[100,100,300],[100,500,300],[80,0,60], [200,1000,300],[10000,10000,300],[700,200,300],[1500,900,300],[80,60,60]])

for _,svbrdf in sample_ds.take(1):
    for i in range(8):
        print('start rendering')
        st = time.time()
        plt.axis('off')
        #maps = (svbrdf[1]+1)/2
        #albedomap, specularmap,roughnessmap, normalinmap= maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6], maps[:,:,7:] 
        #padd1 = tf.ones ([256,256,1],dtype = tf.float32)
        #N = tf.concat([normalinmap,padd1],axis = -1)
        out = GGXtf(svbrdf[0],lightpos_set[i])
        #roughnessmap = tf.reshape(roughnessmap,(256,256,1))
        print('finish rendering, using '+str(time.time()-st)+' seconds')
        #plt.imshow(tf.squeeze(roughnessmap,axis=2),cmap = 'Greys')
        plt.imshow(out)
        plt.show()