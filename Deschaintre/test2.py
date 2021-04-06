import numpy as np
import os
import time
from numpy.lib.function_base import append
import tensorflow as tf
from imageprocessing import imagestack
import matplotlib.pyplot as plt

def normalis(vec):
    return vec/np.linalg.norm(vec)

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXrenderer(maps):
    
    #variable needed are defined below:
    #==========================================================
    imgout = np.zeros([288,288,3])

    #Assuming lightpos at (0,0,10) veiwpos at (0,0,5)
    lightpos = np.array([144,144,1000])
    viewpos  = np.array([144,144,144])
    #assuming original is a square with 288*288, normal is (0,0,1)
    #FragPos = x,y ; texcoords = fragpos
    #F0 = np.array([1,1,1])*0.04
    albedomap, specularmap, normalinmap, roughnessmap = process(maps)

    
    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):
            #for each pixel:
            fragpos = np.array([j,i,0])
            albedo   = albedomap[i,j]/255
            metallic = np.average(specularmap[i,j])/255# set to single value
            alpha_sq= (roughnessmap[i,j]/255)**2
            normaladded = normalinmap[i,j] 
            normaladded /= 255
            N = np.append(normaladded,2)

            N = normalis(N)
            V = normalis(viewpos - fragpos)
            L = normalis(lightpos - fragpos)

            H = normalis(V + L)
            VdotH = np.max(np.dot(V,H),0)
            NdotH = np.max(np.dot(N,H),0)
            NdotV = np.max(np.dot(N,V),0)
            NdotL = np.max(np.dot(N,L),0)
            

            F = metallic+ (1 - metallic)* (1 - VdotH)*5
            NDF = 1 / (np.pi*alpha_sq*pow(NdotH,4.0))*np.exp((NdotH * NdotH - 1.0) / (alpha_sq * NdotH * NdotH))
            G = min(1 ,min( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH))

            nominator    = NDF * G * F 
            denominator = 4 * NdotV * NdotL + 0.001
            specular = nominator / denominator

            #kS = 0.2
            #kD = 1.0 - kS
            #kD *= 1.0 - metallic

            radiance = 1
            #diffuse = kD * albedo / np.pi * radiance *NdotL
            diffuse = (1-metallic) * albedo / np.pi * radiance *NdotL
            reflection = specular * radiance * NdotL
            color =  diffuse + np.ones(3)*reflection 
            imgout[i,j] = 1 if color.any() > 1 else color
            #imgout[i,j] = L

    return imgout

def original(maps):
    imgout = np.zeros([288,288,3])
    lightpos = np.array([144,144,1000])
    viewpos  = np.array([144,144,144])
    albedomap, specularmap, normalinmap, roughnessmap = process(maps)

    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):

            fragpos = np.array([j,i,0])
            albedo   = albedomap[i,j]/255
            metallic = np.average(specularmap[i,j])/255# set to single value
            roughness= roughnessmap[i,j]/255

            normaladded = normalinmap[i,j] 
            normaladded /= 255
            N = np.append(normaladded,1)
            N = normalis(N)

            V = normalis(viewpos - fragpos)
            F0 = 0.2
            L = normalis(lightpos - fragpos)
            H = normalis(V + L)

            NdotH = np.dot(N,H)
            alpha_sq = roughness*roughness
            VdotH = max(np.dot(V,H),0.)
            NdotV = max(np.dot(V,N),0.)
            NdotL = max(np.dot(L,N),0.)

            NDF = 1.0 / (np.pi*alpha_sq*pow(NdotH,4.0))*np.exp((NdotH * NdotH - 1.0) / (alpha_sq * NdotH * NdotH))
            G = min(1.,min(2.*NdotH*NdotV/VdotH,2.*NdotH*NdotL/VdotH))
            F = F0 + (1 - F0) * (1 - VdotH)**5
        
            nominator    = NDF * G * F 
            denominator = 4 * NdotV * NdotL + 0.001
            specular = nominator / denominator

            kS = F
            kD = 1.0 - kS
            kD *= 1.0 - metallic	 

            radiance = 0.5

            Lo = (kD * albedo / np.pi + specular) * radiance * NdotL
            ambient = 0.5 * albedo
            color =  Lo + ambient
            color = color / (color + 1)

            imgout[i,j] = V
    return imgout




path = os.getcwd()+'\Deschaintre\example.png'
_, maps = imagestack(path)
print('start rendering')
st = time.time()

out1 = original(maps)
print('finish rendering, using '+str(time.time()-st)+' seconds')
plt.imshow(out1)
plt.xlabel('H')
plt.show()