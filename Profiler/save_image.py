import tensorflow as tf
from datetime import datetime
import random
from PIL import Image 
import PIL
import cv2




NN_size = 256
batch_size = 8

def outputimg(img):
    shape = 288

    albedo  = img[:,shape*2:shape*3,: ]
    specular= img[:,shape*4:shape*5,: ]
    normal  = img[:,shape  :shape*2,:]
    rough   = img[:,shape*3:shape*4, ]

    '''
    tf.keras.preprocessing.image.save_img('D:\Y4\example\\albedo.png',albedo)
    tf.keras.preprocessing.image.save_img('D:\Y4\example\\specular.png',specular)
    tf.keras.preprocessing.image.save_img('D:\Y4\example\\normal.png',normal)
    tf.keras.preprocessing.image.save_img('D:\Y4\example\\rough.png',rough)
    
    albedo = albedo.save('D:\Y4\example\\albedo.png')
    specular = specular.save('D:\Y4\example\\specular.png')
    normal = normal.save('D:\Y4\example\\normal.png')
    rough = rough.save('D:\Y4\example\\rough.png')
    '''
    cv2.imwrite('D:\Y4\example\\albedo.png',albedo)
    cv2.imwrite('D:\Y4\example\\specular.png',specular)
    cv2.imwrite('D:\Y4\example\\normal.png',normal)
    cv2.imwrite('D:\Y4\example\\rough.png',rough)



path = 'D:\Y4\DNNreimplement\Deschaintre\Dataset\Train_smaller\\0000049;PolishedMarbleFloor_01Xleather_tiles;0X0.png'
img = cv2.imread(path)
outputimg(img)