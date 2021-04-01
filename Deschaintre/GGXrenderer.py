
import numpy as np
import os
import time
from numpy.lib.function_base import append
from imageprocessing import imagestack


def normalisation(vec):
    if np.sum(vec) == 0:
        return vec
    else:
        return vec/np.sum(vec)

def DistributionGGX(N,  H,  roughness):
    a = roughness*roughness
    a2 = a*a
    #print(N,H)
    NdotH = np.max(np.dot(N, H),0)
    NdotH2 = NdotH*NdotH
    nom   = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom

    return nom / denom

def GeometrySchlickGGX(NdotV,  roughness):
    r = (roughness + 1.0)
    k = (r*r) / 8.0
    nom   = NdotV
    denom = NdotV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(N,  V,  L, roughness):
    NdotV = np.max(np.dot(N, V),0)
    NdotL = np.max(np.dot(N, L),0)
    ggx2 = GeometrySchlickGGX(NdotV, roughness)
    ggx1 = GeometrySchlickGGX(NdotL, roughness)

    return ggx1 * ggx2

def fresnelSchlick(cosTheta, F0):
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0)

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXrenderer(maps):
    #using fragment-like shader
    #variable needed are defined below:
    #==========================================================
    imgout = np.zeros([288,288,3])

    #Assuming lightpos at (0,0,10) veiwpos at (0,0,5)
    lightpos = np.array([144,0,10])
    viewpos  = np.array([144,144,5])
    #assuming original is a square with 288*288, normal is (0,0,1)
    normal_global   = np.array([0,0,1])
    #FragPos = x,y ; texcoords = fragpos
    #F0 = np.array([1,1,1])*0.04

    albedomap, specularmap, normalinmap, roughnessmap = process(maps)

    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):
            #for each pixel:
            fragpos = np.array([i,j,0])
            albedo   = albedomap[i,j]
            metallic = np.average(specularmap[i,j])# set to single value
            roughness= roughnessmap[i,j]

            normaladded = normalinmap[i,j] 
            normaladded /= 255
            normal = normal_global + np.append(normaladded,1)

            N = normalisation(normal)
            V = normalisation(viewpos - lightpos)
            #F0 = np.max(F0, albedo, specular)

            L = normalisation(lightpos - fragpos)
            H = normalisation(V + L)

            #print(V,H)
            VdotH = np.max(np.dot(V,H),0)
            #print(VdotH)
            NDF = DistributionGGX(N, H, roughness)  
            G   = GeometrySmith(N, V, L, roughness)  
            #F   = fresnelSchlick(np.max(VdotH, 0), F0)
            F0 = 0.2
            F = F0+ (1 - F0)* (1 - VdotH)*5

            nominator    = NDF * G * F
            denominator = 4 * np.max(np.dot(N, V),0) * np.max(np.dot(N, L),0) + 0.001
            specular = nominator / denominator

            kS = F
            #print(F)
            kD = 1.0 - kS
            kD *= 1.0 - metallic/255

            radiance = 0.5
            #print(albedo, specular)
            
            #print(kD)
            #Lo = (kD * albedo / np.pi ) * radiance *np.max(np.dot(L,N),0)
            Lo = (kD * albedo / np.pi + specular) * radiance *np.max(np.dot(L,N),0)	 
            ambient = 0.5 * albedo
            #print(Lo)
            color = ambient + Lo
            
            #print(color)
            #color /=  color + np.ones(3)*255 #simple tonemapping
            imgout[i,j] = color.astype(int)/255
    
    return imgout


path = os.getcwd()+'\Deschaintre\example.png'
input, maps = imagestack(path)
print('start rendering')
st = time.time()
out = GGXrenderer(maps)
print('finish rendering, using '+str(time.time()-st)+' seconds')
import matplotlib.pyplot as plt
plt.imshow(out)
plt.show()

