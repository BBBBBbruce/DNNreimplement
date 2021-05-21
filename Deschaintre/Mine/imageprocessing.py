#import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os



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

def imagestack_img(img):
    shape = img.shape[0]
    inputimg = img[:,0:shape,:] #rendered 3 channels

    # assemble order:
    #albedo, specular, normal, roughness
    reference = np.zeros([shape,shape,9])
   
    #print(reference[:,:,0:3].shape, img[:,shape*2:shape*3,:].shape)
    reference[:,:,0:3] = img[:,shape*2:shape*3,:]
    reference[:,:,3:6] = img[:,shape*4:shape*5,:]
    reference[:,:,6:8] = img[:,shape:shape*2,1:]
    reference[:,:,8]   = img[:,shape*3:shape*4,0]
    #print(img.shape)
    return inputimg, reference

#=== test ===

path = 'E:\workspace_ms_zhiyuan\DNNreimplement\Deschaintre\Dataset\inputExamples\example.png'
photo, maps = imagestack(path)
display(photo,maps)
#plt.imshow(input)
print (maps.shape)
#np.save(os.getcwd()+'\Deschaintre\\ref',maps)
#numpya = np.load(os.getcwd()+'\Deschaintre\\ref.npy')
#print (numpya.shape)
#plt.show()

