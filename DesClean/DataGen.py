import tensorflow as tf

NN_size = 256
batch_size = 8

def imagestack_img(img):
    shape = img.shape[0]
    inputimg = img[:,0:shape,:] #rendered 3 channels

    albedo  = img[:,shape*2:shape*3,: ]
    specular= img[:,shape*4:shape*5,: ]
    normal  = img[:,shape:shape*2  ,1:]
    rough   = img[:,shape*3:shape*4,0 ]

    return inputimg, tf.concat([albedo,specular,normal,tf.reshape(rough,(shape,shape,1))],axis = -1)

def parse_path(path):
    image_string = tf.io.read_file(path)
    #raw_input = tf.cast(tf.image.decode_image(image_string),tf.float32)
    raw_input = tf.image.decode_image(image_string,dtype = tf.float32)
    #tst = tf.ones((288,288*5,3),dtype = tf.float64)
    return raw_input

def img_process(raw):
    ins,outs = imagestack_img(raw)
    #outs = tf.cast(outs,tf.float32)
    inputs = tf.image.random_crop(ins,  [NN_size, NN_size, 3]) 
    outputs= tf.image.random_crop(outs, [NN_size, NN_size, 9])
    #TODO fix this
    return inputs, outputs

def tf_im_stack_map(raw):
    ins, outs = tf.py_function(func = img_process, inp = [raw],Tout =(tf.float32,tf.float32) )
    ins.set_shape((256,256,3))
    outs.set_shape((256,256,9))
    ins = tf.expand_dims(ins, axis=0)
    outs = tf.expand_dims(outs, axis=0)
    return ins,outs

def svbrdf_gen(path, bs):
    dataset = tf.data.Dataset.list_files(path+'\*.png')
    image_ds = dataset.map(parse_path)
    trainset = image_ds.map(tf_im_stack_map)
    trainset.batch(bs)
    return trainset
















