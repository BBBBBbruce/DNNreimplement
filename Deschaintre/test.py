import numpy as np
import tensorflow as tf

def normalisation(vec):
    return vec/np.linalg.norm(vec)

shapex = 5
shapey = 5

x = np.linspace(0,shapex-1,shapex)
y = np.linspace(0,shapey-1,shapey)
xx,yy = np.meshgrid(x,y)
padd = np.reshape(np.zeros([shapex,shapey]),(5,5,1))
fragpos = np.stack((xx,yy), axis=-1)
frag = np.append(fragpos,padd,axis = -1)

lightpos = np.array([144,144,144])
viewpos = np.array([144,144,128])

L = normalisation(lightpos-frag)
V = normalisation(viewpos-frag)
print(L.shape,V.shape)
LdotV = np.tensordot(L,V,0)
print(LdotV)
