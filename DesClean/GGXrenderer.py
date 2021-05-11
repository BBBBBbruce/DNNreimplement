
import numpy as np
import tensorflow as tf

import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
#
PI = tf.constant(np.pi,dtype = tf.float32)
bs = 8

def imagestack(path):
    image_string = tf.io.read_file(path)
    img = tf.image.decode_image(image_string,dtype = tf.float32)
    shape = img.shape[0]

    inputimg = img[:,0:shape,:] #rendered 3 channels

    albedo  = img[:,shape*2:shape*3,: ]
    specular= img[:,shape*4:shape*5,: ]
    normal  = img[:,shape  :shape*2,:2]
    rough   = img[:,shape*3:shape*4,0 ]
    rough = tf.expand_dims(rough,axis=-1)

    return inputimg, tf.concat([albedo,specular,normal,rough],axis = -1)

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

def rendering_loss(mgt, mif):
    loss = 0
    for i in range(bs):
        gtruth = mgt[i]
        ifred  = mif[i] 
        loss +=  l1_loss(GGXtf(gtruth),GGXtf(ifred)) *0.4+ 0.6* l1_loss(gtruth,ifred)
    return loss/bs

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXtf(maps):
    
    def match_dim (map):
        return tf.expand_dims(map,axis = -1)

    def GGXpxl(V,L,N,albedo,metallic,rough):
        #print(V.shape,L.shape,N.shape,albedo.shape,rough.shape)
        #metallic = tf.reduce_mean(metallic,axis = -1)# set to single value
        rough= rough**2
        rough = match_dim(rough)
        H = normalisation(V+L)
        VdotH = tf.maximum(tf.reduce_sum(V*H,axis = -1,keepdims = True),0)
        NdotH = tf.maximum(tf.reduce_sum(N*H,axis = -1,keepdims = True),0)
        NdotV = tf.maximum(tf.reduce_sum(V*N,axis = -1,keepdims = True),0)
        NdotL = tf.maximum(tf.reduce_sum(N*L,axis = -1,keepdims = True),0)

        F = metallic+ tf.math.multiply((1 - metallic) , (1 - VdotH)**5)
        NDF = 1 / (PI*rough*pow(NdotH,4.0))*tf.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
        G = tf.minimum( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH)
        G = tf.minimum(tf.cast(1,dtype = tf.float32) , G)
        nominator    = NDF* G * F 
        denominator = 4 * NdotV * NdotL + 0.001
        specular = nominator / denominator
        
        #diffuse = (1-metallic)[:,:,None] * albedo / PI *NdotL[:,:,None] 

        diffuse = (1-metallic) * albedo / PI *NdotL

        reflection = specular * NdotL*4 #* radiance 
        #reflection = tf.reshape(reflection,(256,256,1))

        #color = tf.concat([reflection,reflection,reflection],-1) + diffuse*1
        color = reflection + diffuse*1
        return color**(1/1.8)

    maps = tf.squeeze(maps)
    lightpos = tf.constant([288,288,200],dtype = tf.float32)
    viewpos  = tf.constant([143,143,288],dtype = tf.float32)

    albedomap, specularmap, normalinmap, roughnessmap = process(maps)
    
    
    shapex = 256
    shapey = 256

    x = np.linspace(0,shapex-1,shapex)
    y = np.linspace(0,shapey-1,shapey)
    xx,yy = tf.meshgrid(x,y)
    xx = tf.cast(tf.reshape(xx ,(shapex,shapey,1)),dtype = tf.float32)
    yy = tf.cast(tf.reshape(yy ,(shapex,shapey,1)),dtype = tf.float32)
    padd0 = tf.reshape(tf.zeros([shapex,shapey],dtype = tf.float32)    ,(shapex,shapey,1))
    padd1 = tf.reshape(tf.ones ([shapex,shapey],dtype = tf.float32),(shapex,shapey,1))
    fragpos = tf.concat([xx,yy,padd0],axis = -1)

    N = normalisation(tf.concat([normalinmap,padd1],axis = -1))
    V = normalisation(viewpos - fragpos)
    L = normalisation(lightpos - fragpos)
    '''
    rough = tf.expand_dims(roughnessmap,axis=-1)
    title = ['view','light','albedo', 'specular', 'normal','roughness']
    display_list=[V,L, albedomap, specularmap, N, rough]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    '''

    imgout = GGXpxl(V ,L , N, albedomap,specularmap,roughnessmap)
    return  imgout


'''
tf.compat.v1.enable_eager_execution()

path = 'E:\workspace_ms_zhiyuan\DNNreimplement\Deschaintre\Dataset\inputExamples\example.png'
photo, maps = imagestack(path)

ran_seed = random.seed(datetime.now())
tf.random.set_seed(ran_seed) 
maps= tf.image.random_crop(maps, [256, 256, 9])

print('start rendering')
st = time.time()
out1 = GGXtf(maps)

print('finish rendering, using '+str(time.time()-st)+' seconds')
plt.imshow(out1)
plt.show()
'''