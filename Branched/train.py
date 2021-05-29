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

def train_split(mode = 'all'):
    if (mode == 'a'):
        print('***************** TRAINING ALBEDO NET ************************')
        model_a = net_split(filter_3,3)

        print('load_data')
        ds = svbrdf_gen(train_path,8,'a')
        test_ds = svbrdf_gen(test_path,8,'a')
        print('finish_loading')

        model_a.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        hitory = model_a.fit( ds,verbose =1 , steps_per_epoch = step_num, epochs=epoch_num)#,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

        model_a.save('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_alebdo_1')
        new_model_a = tf.keras.models.load_model('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_alebdo_1', custom_objects = {'rendering_loss' : rendering_loss},compile=False )

        new_model_a.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        oss, acc = new_model_a.evaluate(test_ds, verbose=2,steps=10)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
        
    elif (mode == 's'):
        print('***************** TRAINING SPECULAR NET ************************')
        model_s = net_split(filter_3,3)

        print('load_data')
        ds = svbrdf_gen(test_path,8,'s')
        test_ds = svbrdf_gen(test_path,8,'s')
        print('finish_loading')

        model_s.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        hitory = model_s.fit( ds,verbose =1 , steps_per_epoch = step_num, epochs=epoch_num)#,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

        model_s.save('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_specular_1')
        new_model_s = tf.keras.models.load_model('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_specular_1', custom_objects = {'rendering_loss' : rendering_loss},compile=False )

        new_model_s.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        oss, acc = new_model_s.evaluate(test_ds, verbose=2,steps=10)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    
    elif (mode == 'n'):
        print('***************** TRAINING NORMAL NET ************************')
        model_n = net_split(filter_2,2)

        print('load_data')
        ds = svbrdf_gen(test_path,8,'n')
        test_ds = svbrdf_gen(test_path,8,'n')
        print('finish_loading')

        model_n.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        hitory = model_n.fit( ds,verbose =1 , steps_per_epoch = step_num, epochs=epoch_num)#,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

        model_n.save('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_normal_1')
        new_model_n = tf.keras.models.load_model('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_normal_1', custom_objects = {'rendering_loss' : rendering_loss},compile=False )

        new_model_n.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        oss, acc = new_model_n.evaluate(test_ds, verbose=2,steps=10)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    
    elif (mode == 'r'):
        print('***************** TRAINING ROUGHNESS NET ************************')
        model_r = net_split(filter_1,1)

        print('load_data')
        ds = svbrdf_gen(test_path,8,'r')
        test_ds = svbrdf_gen(test_path,8,'r')
        print('finish_loading')

        model_r.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        hitory = model_r.fit( ds,verbose =1 , steps_per_epoch = step_num, epochs=epoch_num)#,callbacks=[tensorboard_callback]) #24884 DisplayCallback()

        model_r.save('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_roughness_1')
        new_model_r = tf.keras.models.load_model('/vol/bitbucket/zz6117/DNNreimplement/Model_trained/Model_saved_roughness_1', custom_objects = {'rendering_loss' : rendering_loss},compile=False )

        new_model_r.compile(optimizer = opt, loss = rendering_loss, metrics = ['accuracy'])
        oss, acc = new_model_r.evaluate(test_ds, verbose=2,steps=10)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    elif (mode == 'all'):
        print('***************** SEQUENTIALLY TRAINING ALL NETS ************************')
        train_split('a')
        train_split('s')
        train_split('n')
        train_split('r')

    else:
        print('dude, wtf is this')

#print(sys.argv[1])
train_split(sys.argv[1])