
import numpy as np
import tensorflow as tf

PI = tf.constant(np.pi,dtype = tf.float32)
bs = 8

viewpos  = tf.constant([143,143,288],dtype = tf.float32)

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]


def padd1_fragepos():
    shapex = 256
    shapey = 256
    x = np.linspace(0,shapex-1,shapex)
    y = np.linspace(0,shapey-1,shapey)
    xx,yy = tf.meshgrid(x,y)
    xx = tf.cast(tf.reshape(xx ,(shapex,shapey,1)),dtype = tf.float32)
    yy = tf.cast(tf.reshape(yy ,(shapex,shapey,1)),dtype = tf.float32)
    padd0 = tf.zeros([shapex,shapey,1],dtype = tf.float32)
    padd1 = tf.ones ([shapex,shapey,1],dtype = tf.float32)
    fragpos = tf.concat([xx,yy,padd0],axis = -1)
    return fragpos, padd1

fragpos,padd1 = padd1_fragepos()
V = normalisation(viewpos - fragpos)

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

def rendering_loss(mgt, mif):
    loss = 0
    for i in range(bs):
        gtruth = mgt[i]
        ifred  = mif[i] 
        loss +=  l1_loss(GGXtf(gtruth),GGXtf(ifred)) #*0.4 + 0.6* l1_loss(gtruth,ifred)
        loss +=  l1_loss(GGXtf(gtruth),GGXtf(ifred))
        loss +=  l1_loss(GGXtf(gtruth),GGXtf(ifred))
        loss +=  l1_loss(GGXtf(gtruth),GGXtf(ifred))
        loss /= 4
    return loss/bs

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXtf(maps):
    
    def match_dim (map):
        return tf.expand_dims(map,axis = -1)

    def GGXpxl(V,L,N,albedo,metallic,rough):
        rough= rough**2
        rough = match_dim(rough+0.01)
        H = normalisation(V+L)
        VdotH = tf.maximum(tf.reduce_sum(V*H,axis = -1,keepdims = True),0.01)
        NdotH = tf.maximum(tf.reduce_sum(N*H,axis = -1,keepdims = True),0.01)
        NdotV = tf.maximum(tf.reduce_sum(V*N,axis = -1,keepdims = True),0.01)
        NdotL = tf.maximum(tf.reduce_sum(N*L,axis = -1,keepdims = True),0.01)

        F = metallic+ tf.math.multiply((1 - metallic) , (1 - VdotH)**5)
        NDF = 1 / (PI*rough*pow(NdotH,4.0))*tf.exp((NdotH * NdotH - 1.0) / (rough * NdotH * NdotH))
        G = tf.minimum( 2*NdotH*NdotV/VdotH, 2*NdotH*NdotL/VdotH)
        G = tf.minimum(1.0 , G)
        nominator    = NDF* G * F 
        denominator = 4 * NdotV * NdotL +0.01
        specular = nominator / denominator
        
        diffuse = (1-metallic) * albedo / PI *NdotL

        reflection = specular * NdotL*1.5 #* radiance 

        color = reflection + diffuse
        return color#**(1/1.8)

    maps = tf.squeeze(maps)
    maps = (maps+1)/2

    np.random.seed() # random takes long time
    xy = np.random.randint(288,10000, size=2)
    lightpos = tf.constant([xy[0],xy[1],1000],dtype = tf.float32)
    

    albedomap, specularmap, normalinmap, roughnessmap = process(maps)
    
    N = normalisation(tf.concat([normalinmap,padd1],axis = -1))
    L = normalisation(lightpos - fragpos)

    imgout = GGXpxl(V ,L , N, albedomap,specularmap,roughnessmap)
    return  imgout