import numpy as np
import tensorflow as tf

def normalisation(vec):
    return vec/np.linalg.norm(vec,axis = -1)[:,:,None]

shapex = 5
shapey = 5

x = np.linspace(0,shapex-1,shapex)
y = np.linspace(0,shapey-1,shapey)
xx,yy = tf.convert_to_tensor(np.meshgrid(x,y))
#X = tf.stack(tf.Variable([1,2]),tf.Variable(2),1)

padd = np.reshape(np.zeros([shapex,shapey]),(5,5,1))
fragpos = np.stack((xx,yy), axis=-1)

frag = np.append(fragpos,padd,axis = -1)

lightpos = np.array([5,5,-6])
viewpos = np.array([0,0,10])

print(frag.shape)
L = normalisation(lightpos-frag)
V = normalisation(viewpos-frag)

print(V)

lDOTv = tf.maximum(tf.reduce_sum(L * V, axis=-1),0)

print(lDOTv)
