
from datagen import svbrdf_gen
from loss import rendering_loss
import time


sample = 'D:\Y4\DNNreimplement\Deschaintre\Dataset\Train_smaller'
sample_ds = svbrdf_gen(sample,8)

for _, svbrdf1 in sample_ds.take(1):
    for photo, svbrdf2 in sample_ds.take(1):
        st = time.time()
        print(rendering_loss(svbrdf1,svbrdf2))
        print('finish rendering, using '+str(time.time()-st)+' seconds')
        #st = time.time()
        #print(rendering_loss2(svbrdf1,svbrdf2))
        #print('finish rendering, using '+str(time.time()-st)+' seconds')