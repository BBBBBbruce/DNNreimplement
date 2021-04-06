import numpy as np
import tensorflow as tf

def normalisation(vec):
    #return vec/np.linalg.norm(vec)
    return vec/np.linalg.norm(vec,axis = -1)[:,:,None]

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

shapex = 5
shapey = 5

x = np.linspace(0,shapex-1,shapex)
y = np.linspace(0,shapey-1,shapey)
xx,yy = tf.convert_to_tensor(np.meshgrid(x,y))
#X = tf.stack(tf.Variable([1,2]),tf.Variable(2),1)

padd = np.reshape(np.zeros([shapex,shapey]),(5,5,1))
fragpos = np.stack((xx,yy), axis=-1)

frag = np.append(fragpos,padd,axis = -1)

lightpos = np.array([0,0,10000])
viewpos = np.array([0,0,10])

#print(frag.shape)
L = normalisation(lightpos-frag)
V = normalisation(viewpos-frag)

#L = np.array([-144,-144,144])
print(normalisation(L))

lDOTv = tf.maximum(tf.reduce_sum(L * V, axis=-1),0)

#print(lDOTv)


def GGX(L, V, N, albedo, metallic, rough):
    #per pixel
    H = normalisation(V + L)
    VdotH = np.max(np.dot(V,H),0)
    NdotH = np.max(np.dot(N,H),0)
    NdotV = np.max(np.dot(N,V),0)
    NdotL = np.max(np.dot(N,L),0)
    F = metallic+ (1 - metallic)* (1 - VdotH)*5
    NDF = 1 / (np.pi*rough*pow(NdotH,4.0))*np.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
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

    #color /=  color + np.ones(3) #simple tonemapping
    return 1 if color.any() > 1 else color


def GGXrendererOLD(maps):

    #variable needed are defined below:
    #==========================================================
    imgout = np.zeros([288,288,3])
    #Assuming lightpos at (0,0,10) veiwpos at (0,0,5)
    lightpos = np.array([200,200,1000])
    viewpos  = np.array([144,144,144])
    #assuming original is a square with 288*288, normal is (0,0,1)
    normal_global   = np.array([0,0,1])
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
            normal = normal_global + np.append(normaladded,1)
            N = normalisation(normal)
            V = normalisation(viewpos - fragpos)
            L = normalisation(lightpos - fragpos)
            imgout[i,j] = GGX(L,V,N,albedo, metallic,alpha_sq)
    return imgout