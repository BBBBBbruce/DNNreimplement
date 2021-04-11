import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def imagestack(path):
    img = np.array(cv.imread(path))
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
'''
path = os.getcwd()+'\Deschaintre\example.png'
input, maps = imagestack(path)

#plt.imshow(input)
print (maps.shape)
np.save(os.getcwd()+'\Deschaintre\\ref',maps)
numpya = np.load(os.getcwd()+'\Deschaintre\\ref.npy')
print (numpya.shape)
#plt.show()
'''
