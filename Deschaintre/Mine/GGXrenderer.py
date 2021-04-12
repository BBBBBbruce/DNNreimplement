
import numpy as np
import os
import time
from numpy.lib.function_base import append
import tensorflow as tf
from imageprocessing import imagestack
import matplotlib.pyplot as plt




def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

def normalis(vec):
        return vec/np.linalg.norm(vec)

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

def GeometrySchlickGGX(NdotV,  k):
    return 1.0/tf.maximum((NdotV * (1.0 - k) + k), 0.001)

def GeometrySmith(NdotV, NdotL, roughness):
    
    ggx2 = GeometrySchlickGGX(NdotV, tf.sqrt(roughness))
    ggx1 = GeometrySchlickGGX(NdotL, tf.sqrt(roughness))
    return ggx1 * ggx2

def fresnelSchlick(cosTheta, F0):
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0)

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def reflect(I,N):
    #return 2*np.dot(dir,N)*N - dir
    return I - 2 * np.dot(I, N) * N

'''
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
'''

def GGXpxl(V,L,N,albedo,metallic,rough):
        #print(L)
        
        albedo   = albedo/255
        metallic = tf.reduce_mean(metallic,axis = -1)/255# set to single value
        rough= (rough/255)**2

        #H = normalisation(V + L)
        H = normalisation(V+L)
        
        VdotH = tf.maximum(tf.reduce_sum(V*H,axis = -1),0)
        NdotH = tf.maximum(tf.reduce_sum(N*H,axis = -1),0)
        NdotV = tf.maximum(tf.reduce_sum(V*N,axis = -1),0)
        NdotL = tf.maximum(tf.reduce_sum(N*L,axis = -1),0)
        #print(NdotH[0,0],NdotH[287,287],NdotH[0,287],NdotH[287,0])
        #VdotH = tf.reduce_sum(V*H,axis = -1)
        #NdotH = tf.reduce_sum(N*H,axis = -1)
        #NdotV = tf.reduce_sum(V*N,axis = -1)
        #NdotL = tf.reduce_sum(N*L,axis = -1)


        F = metallic+ (1 - metallic) * (1 - VdotH)**5
        NDF = 1 / (np.pi*rough*pow(NdotH,4.0))*tf.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
        G = tf.minimum( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH)
        G = tf.minimum(tf.cast(1,dtype = tf.float64) , G)
        #G = GeometrySmith(NdotV, NdotL, rough)

        nominator    = NDF* G * F 
        denominator = 4 * NdotV * NdotL + 0.001
        specular = nominator / denominator

        #radiance = 1
        #diffuse = kD * albedo / tf.pi * radiance *NdotL
        diffuse = (1-metallic)[:,:,None] * albedo / np.pi *NdotL[:,:,None] #*radiance

        reflection = specular * NdotL*4 #* radiance 
        reflection = tf.reshape(reflection,(288,288,1))
        #print(tf.concat([reflection,reflection,reflection],-1).shape)
        color = tf.concat([reflection,reflection,reflection],-1) + diffuse*1
        return tf.minimum(tf.cast(1,dtype = tf.float64),color)

def GGXtf(maps):
    lightpos = tf.Variable([100,200,200],dtype = tf.float64)
    viewpos  = tf.Variable([143,143,288],dtype = tf.float64)
    #assuming original is a square with 288*288, normal is (0,0,1)
    #normal_global   = np.array([0,0,1]) # depth map
    #FragPos = x,y ; texcoords = fragpos
    #F0 = np.array([1,1,1])*0.04
    albedomap, specularmap, normalinmap, roughnessmap = process(maps)
    
    shapex = int(maps.shape[0])
    shapey = int(maps.shape[1])
    #print(shapex,shapey)
    x = np.linspace(0,shapex-1,shapex)
    y = np.linspace(0,shapey-1,shapey)
    xx,yy = tf.meshgrid(x,y)
    xx = tf.reshape(xx ,(shapex,shapey,1))
    yy = tf.reshape(yy ,(shapex,shapey,1))
    padd0 = tf.reshape(tf.zeros([shapex,shapey],dtype = tf.float64)    ,(shapex,shapey,1))
    padd1 = tf.reshape(tf.ones ([shapex,shapey],dtype = tf.float64)*255,(shapex,shapey,1))
    fragpos = tf.concat([xx,yy,padd0],axis = -1)
    #fragpos = np.append(np.stack((xx,yy), axis=-1),padd0,axis = -1)
    #N = normalisation(tf.concat([normalinmap,padd1],axis = -1)/255)
    N = normalisation(tf.concat([normalinmap,padd1],axis = -1)/255)
    V = normalisation(viewpos - fragpos)
    L = normalisation(lightpos - fragpos)
    #print(L)
    #N = normalisation(N)
    #print(V[0,0],V[287,287], V[287,0], V[0,287])
    imgout = GGXpxl(V ,L , N, albedomap,specularmap,roughnessmap)

    return  imgout

def GGXperpixel(maps):
    def GGXpxl(x,y,normal,albedo,metallic,rough):
        fragpos = np.array([x,y,0])
        albedo   = albedo/255
        metallic = np.average(metallic)/255# set to single value
        rough= (rough/255)**2
        normaladded = normal/255

        #N = normalisation(normal_g + np.append(normaladded,1))
        N = normalisation(np.append(normaladded,2))
        V = normalisation(viewpos - fragpos)
        L = normalisation(lightpos - fragpos)
        H = normalisation(V + L)
        VdotH = np.max(np.dot(V,H),0)
        NdotH = np.max(np.dot(N,H),0)
        NdotV = np.max(np.dot(N,V),0)
        NdotL = np.max(np.dot(N,L),0)

        F = metallic+ (1 - metallic)* (1 - VdotH)*5
        NDF = 1 / (np.pi*rough*pow(NdotH,4.0))*np.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
        G = min(1. ,min( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH))

        nominator    = NDF * G * F 
        denominator = 4 * NdotV * NdotL + 0.001
        specular = nominator / denominator

        radiance = 1
        #diffuse = kD * albedo / np.pi * radiance *NdotL
        diffuse = (1-metallic) * albedo / np.pi * radiance *NdotL
        reflection = specular * radiance * NdotL
        color =  diffuse + np.ones(3)*reflection 

        return 1 if color.any() > 1 else color#**(1/1.5) 

    lightpos = np.array([0,0,1000])
    viewpos  = np.array([144,144,144])
    #assuming original is a square with 288*288, normal is (0,0,1)
    #normal_global   = np.array([0,0,1]) # depth map
    #FragPos = x,y ; texcoords = fragpos
    #F0 = np.array([1,1,1])*0.04
    albedomap, specularmap, normalinmap, roughnessmap = process(maps)
    x = np.linspace(0,maps.shape[0]-1,maps.shape[0])
    y = np.linspace(0,maps.shape[1]-1,maps.shape[1])
    xx,yy = np.meshgrid(x,y)
    padd = np.reshape(np.zeros([maps.shape[0],maps.shape[1]]),(maps.shape[0],maps.shape[1],1))
    #fragpos = np.append(np.stack((xx,yy), axis=-1),padd,axis = -1)
    #V = normalisation(viewpos - fragpos)
    #L = normalisation(lightpos - fragpos)

    imgout = np.zeros([288,288,3])
    #imgout = tf.while_loop(condtion , GGXpxl, [xx,yy, normalinmap, albedomap,specularmap,roughnessmap] )

    
    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):
            imgout[i,j] = GGXpxl(xx[i,j],yy[i,j], normalinmap[i,j], albedomap[i,j],specularmap[i,j],roughnessmap[i,j])
    
    #imgout = GGXpxl(V,L, normalinmap, albedomap,specularmap,roughnessmap)
    return imgout

def GGX(L, V, N, albedo, metallic, rough):
    #per pixel
    H = normalis(V + L)
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

    return 1 if color.any() > 1 else color#**(1/1.5) 

def GGXrenderer(maps):
    
    #variable needed are defined below:
    #==========================================================
    imgout = np.zeros([288,288,3])

    #Assuming lightpos at (0,0,10) veiwpos at (0,0,5)
    lightpos = np.array([144,144,1000])
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
            N = np.append(normaladded,2)

            N = normalis(N)
            V = normalis(viewpos - fragpos)
            L = normalis(lightpos - fragpos)
            imgout[i,j] = GGX(L,V,N,albedo, metallic,alpha_sq)
            #imgout[i,j] = L
    return imgout

def Phongrenderer(maps):
    #using fragment-like shader
    #variable needed are defined below:
    #==========================================================
    imgout = np.zeros([288,288,3])

    #Assuming lightpos at (0,0,10) veiwpos at (0,0,5)
    lightpos = np.array([144,144,144])
    viewpos  = np.array([144,144,5])
    #assuming original is a square with 288*288, normal is (0,0,1)
    normal_global   = np.array([0,0,1])
    #FragPos = x,y ; texcoords = fragpos
    #F0 = np.array([1,1,1])*0.04

    albedomap, specularmap, normalinmap, roughnessmap = process(maps)

    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):
            #for each pixel:
            fragpos = np.array([j,i,0])
            albedo   = albedomap[i,j]
            #print(albedo)
            metallic = np.average(specularmap[i,j])# set to single value
            #roughness= roughnessmap[i,j]

            normaladded = normalinmap[i,j] 
            normaladded /= 255
            normal = normal_global + np.append(normaladded,1)

            N = normalisation(normal)
            V = normalisation(viewpos - fragpos)
            #print(V)
            #V = viewpos - fragpos
            L = normalisation(lightpos - fragpos)
            #F0 = np.max(F0, albedo, specular)

            ambient = albedo*0.3
            diffuse = albedo* np.dot(N,L)
            R = normalisation( reflect(-L,N))
            spec = np.dot(V,R)**32
            specular = spec * metallic

            
            #print(color)
            #x = 10 if a > b else 11
            c = ambient + diffuse #+ specular
            c /=  c + np.ones(3)*255 #simple tonemapping
            #c2 = ambient + diffuse + specular
            #c2 /=  c2 + np.ones(3)*255 #simple tonemapping
            #print(c1- c2)
            imgout[i,j] = R
    
    return imgout


tf.compat.v1.enable_eager_execution()
print(os.getcwd())
path = os.getcwd()+'\example.png'
#path = os.getcwd()+'\Deschaintre\example.png'
input, maps = imagestack(path)
print('start rendering')
st = time.time()
out1 = GGXtf(tf.convert_to_tensor(maps))

print('finish rendering, using '+str(time.time()-st)+' seconds')
plt.imshow(out1)
plt.show()

#plt.imshow(out2)
#plt.show()


#plt.imshow(out)
#plt.show()

