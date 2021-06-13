import numpy as np
import tensorflow as tf

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

PI = tf.constant(np.pi,dtype = tf.float32)
bs = 8
shape = (256,256,1)
viewpos  = tf.constant([0,0,500],dtype = tf.float32)
padd0 = tf.zeros(shape,dtype = tf.float32) 
padd1 = tf.ones((8,256,256,1),dtype = tf.float32) 

def normalisation(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keepdims=True))
    return tf.math.truediv(tensor, Length)
    
def fragepos():
    x = np.linspace(-128,127,256)
    y = np.linspace(127,-128,256)
    xx,yy = tf.meshgrid(y,x)
    xx = tf.cast(tf.reshape(xx ,shape),dtype = tf.float32)
    yy = tf.cast(tf.reshape(yy ,shape),dtype = tf.float32)
    fragpos = tf.concat([xx,yy,padd0],axis = -1)
    return fragpos

fragpos = fragepos()
V = normalisation(viewpos - fragpos)
V8 = np.array([V,V,V,V,V,V,V,V])

L8 = np.array([normalisation(tf.constant([500,700,300],dtype = tf.float32) - fragpos),
              normalisation(tf.constant([700,0,300],dtype = tf.float32) - fragpos), 
              normalisation(tf.constant([900,700,300],dtype = tf.float32) - fragpos), 
              normalisation(tf.constant([2000,2000,1000],dtype = tf.float32) - fragpos), 
              normalisation(tf.constant([2000,0,2000],dtype = tf.float32) - fragpos),
              normalisation(tf.constant([800,400,800],dtype = tf.float32) - fragpos),
              normalisation(tf.constant([6000,6000,400],dtype = tf.float32) - fragpos), 
              normalisation(tf.constant([70,70,200],dtype = tf.float32) - fragpos) 
              ])

H8 = normalisation(L8 + V8)
VdotH8 = tf.maximum(tf.reduce_sum(V8*H8,axis = -1,keepdims = True),0.01)

def GGX_(maps,V,L,H,VdotH):
    def match_dim (map):
        return tf.expand_dims(map,axis = -1)
    def G_GGX(n):
        return 2*n / (n+(rough**2+(1-rough**2)*n**2)**0.5)
    def process(maps):
        return maps[:,:,:,0:3], maps[:,:,:,3:6], maps[:,:,:,6], maps[:,:,:,7:] 
        
    maps = tf.squeeze(maps)
    maps = (maps+1)/2   
    albedo, metallic,rough, normalinmap= process(maps)
    N = normalisation(tf.concat([normalinmap,padd1],axis = -1))
    rough= rough**2
    rough = match_dim(rough+0.01)

    NdotH = tf.maximum(tf.reduce_sum(N*H,axis = -1,keepdims = True),0.01)
    NdotV = tf.maximum(tf.reduce_sum(V*N,axis = -1,keepdims = True),0.01)
    NdotL = tf.maximum(tf.reduce_sum(N*L,axis = -1,keepdims = True),0.01)

    F = metallic+ tf.math.multiply((1 - metallic) , (1 - VdotH)**5)
    NDF = rough / PI / pow(NdotH*NdotH *(rough-1)+1,2)
    G = tf.multiply(G_GGX(NdotL),G_GGX(NdotV))

    nominator    = NDF* G * F 
    denominator = 4 * NdotV * NdotL +0.01
    specular = nominator / denominator
    
    diffuse = (1-metallic) * albedo / PI *NdotL
    reflection = specular * NdotL*1.5 #* radiance 
    color = reflection + diffuse
    return color**(1/1.8)
    
def render_loss(mgt, mif):
    return l1_loss(GGX_(mgt,V8,L8,H8,VdotH8 ),GGX_(mif,V8,L8,H8,VdotH8 ))