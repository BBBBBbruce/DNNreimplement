import numpy as np
import tensorflow as tf

PI = tf.constant(np.pi,dtype = tf.float32)
bs = 8
shape = (256,256,1)
viewpos  = tf.constant([0,0,500],dtype = tf.float32)
padd0 = tf.zeros(shape,dtype = tf.float32) 
padd1 = tf.ones(shape,dtype = tf.float32) 
lightpos_set = np.array([[400,800,300],[800,400,300],[550,550,300], [200,200,300],[2000,2000,300]])

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

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

def l1_loss(mgt, mif):
    return tf.reduce_mean(tf.abs(mgt-mif))

def l2_loss(mgt, mif):
    return tf.reduce_mean(tf.square(mgt-mif))

def rendering_loss(mgt, mif):
    loss = 0
    for i in range(bs):
        gtruth = mgt[i]
        ifred  = mif[i] 
        loss +=  l1_loss(GGXtf(gtruth,lightpos_set[0]),GGXtf(ifred,lightpos_set[0])) #*0.4 + 0.6* l1_loss(gtruth,ifred)
        loss +=  l1_loss(GGXtf(gtruth,lightpos_set[1]),GGXtf(ifred,lightpos_set[1]))
        loss +=  l1_loss(GGXtf(gtruth,lightpos_set[2]),GGXtf(ifred,lightpos_set[2]))
        loss +=  l1_loss(GGXtf(gtruth,lightpos_set[3]),GGXtf(ifred,lightpos_set[3]))
        loss +=  l1_loss(GGXtf(gtruth,lightpos_set[4]),GGXtf(ifred,lightpos_set[4]))
        loss /= 4
    return loss/bs

def normalisation(vec):
    return vec/tf.norm(vec,axis = -1)[:,:,None]

def process(maps):
    return maps[:,:,0:3], maps[:,:,3:6], maps[:,:,6:8], maps[:,:,8] 

def GGXtf(maps,lightpos):
    def match_dim (map):
        return tf.expand_dims(map,axis = -1)

    def GGXpxl(V,L,N,albedo,metallic,rough):
        def G_GGX(n):
            return 2*n / (n+(rough**2+(1-rough**2)*n**2)**0.5)

        rough= rough**2
        rough = match_dim(rough+0.01)
        H = normalisation(V+L)
        VdotH = tf.maximum(tf.reduce_sum(V*H,axis = -1,keepdims = True),0.01)
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
        color = reflection #+ diffuse
        return color#**(1/1.8)

    lightpos = tf.constant(lightpos,dtype = tf.float32)
    maps = tf.squeeze(maps)
    maps = (maps+1)/2   
    albedomap, specularmap, normalinmap, roughnessmap = process(maps)
    
    N = normalisation(tf.concat([normalinmap,padd1],axis = -1))
    L = normalisation(lightpos - fragpos)
    imgout = GGXpxl(V ,L , N, albedomap,specularmap,roughnessmap)
    return  imgout