
import numpy as np
import tensorflow as tf
#
PI = tf.constant(np.pi,dtype = tf.float32)
bs = 8

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

def rendering_loss(mgt, mif):
    loss = 0
    for i in range(bs):
        gtruth = mgt[i]
        ifred  = mif[i] 
        loss += 0.4* l1_loss(GGXtf(gtruth),GGXtf(ifred)) + 0.6* l1_loss(gtruth,ifred)
    return loss

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXtf(maps):
    
    def GGXpxl(V,L,N,albedo,metallic,rough):
        albedo   = albedo/255
        metallic = tf.reduce_mean(metallic,axis = -1)/255# set to single value
        rough= (rough/255)**2
        H = normalisation(V+L)
        VdotH = tf.maximum(tf.reduce_sum(V*H,axis = -1),0)
        NdotH = tf.maximum(tf.reduce_sum(N*H,axis = -1),0)
        NdotV = tf.maximum(tf.reduce_sum(V*N,axis = -1),0)
        NdotL = tf.maximum(tf.reduce_sum(N*L,axis = -1),0)
        F = metallic+ (1 - metallic) * (1 - VdotH)**5
        NDF = 1 / (PI*rough*pow(NdotH,4.0))*tf.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
        G = tf.minimum( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH)
        G = tf.minimum(tf.cast(1,dtype = tf.float32) , G)
        nominator    = NDF* G * F 
        denominator = 4 * NdotV * NdotL + 0.001
        specular = nominator / denominator
        diffuse = (1-metallic)[:,:,None] * albedo / PI *NdotL[:,:,None] 

        reflection = specular * NdotL*4 #* radiance 
        reflection = tf.reshape(reflection,(256,256,1))
        color = tf.concat([reflection,reflection,reflection],-1) + diffuse*1
        return tf.minimum(tf.cast(1,dtype = tf.float32),color)

    maps = tf.squeeze(maps)
    lightpos = tf.constant([100,200,200],dtype = tf.float32)
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
    padd1 = tf.reshape(tf.ones ([shapex,shapey],dtype = tf.float32)*255,(shapex,shapey,1))
    fragpos = tf.concat([xx,yy,padd0],axis = -1)

    N = normalisation(tf.concat([normalinmap,padd1],axis = -1)/255)
    V = normalisation(viewpos - fragpos)
    L = normalisation(lightpos - fragpos)

    imgout = GGXpxl(V ,L , N, albedomap,specularmap,roughnessmap)
    return  imgout



