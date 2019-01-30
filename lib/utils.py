import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.misc
from glob import glob
from PIL import Image
import tensorflow as tf
#from skimage.transform import resize


def to_rgb(images):
    stacked_img = np.stack((images,)*3, -1)
    stacked_img=np.squeeze(stacked_img)
    return stacked_img

#def to_32(x):
#    x=resize(x,[x.shape[0],32,32,3])
#    return x

def grid_x(X):
    # [-1, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = ((X + 1.) * 127.5).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    return img.astype('uint8')

def normalize(x) :
    return x/127.5 - 1

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, merge(images, size))

def show_digits(digits_flatten,size=28):
    X = grid_x(digits_flatten)
    plt.imshow(X)
    plt.axis('off')
    plt.show()
    
def show_digit(digits_flatten,size=28):
    X = grid_x(digits_flatten)
    plt.imshow(X)
    plt.axis('off')

def read_omniglot(path,size,convert='RGB'):
    #/home/yazici/Documents/data/omniglot-master/python/images_background
    languages = ['Cyrillic','Greek','Latin']
    trainx = []
    trainy = []
    for idx, language in enumerate(languages):
        im_path = glob(path+'/'+language+'/*/*.png')
        for im in im_path:
            im = Image.open(im)
            if convert == 'RGB':
                im = im.convert('RGB')
            elif convert == 'grayscale':
                im = im.convert('L')
            im = im.resize((size,size),Image.BILINEAR)
            im = np.asarray(im,dtype='float32')
            im = -1*im +255
            trainx.append(im)
            trainy.append(idx)
    
    return np.stack(trainx), np.stack(trainy)

def read_comnist(path,size,convert='RGB'):
    #/home/yazici/Documents/data/CoMNIST-master/images
    languages = ['Cyrillic','Latin']
    trainx = []
    trainy = []
    for idx, language in enumerate(languages):
        im_path = glob(path+'/'+language+'/*/*.png')
        for im in im_path:
            im = Image.open(im)
            im = im.resize((size,size),Image.BILINEAR)
            im = np.asarray(im,dtype='float32')
            im = im[:,:,3]
            if convert == 'RGB':
                im = np.stack([im,im,im],axis=2)
            #elif convert == 'grayscale':
            #    im = im.convert('RGB').convert('L')
            
            trainx.append(im)
            trainy.append(idx)
    
    return np.stack(trainx), np.stack(trainy)

def parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
    image_decoded.set_shape((64, 64, 3))
    image_decoded = tf.cast(image_decoded , tf.float32) /127.5 -1
    #image_decoded = tf.image.convert_image_dtype(image_decoded,tf.float32)
    #image_resized = tf.image.resize_images(image_decoded, [size, size])
    return image_decoded#tf.add(tf.multiply(image_decoded,2), -1)

def exponential_moving_average(target_var,source_var,beta):
    ema_op =[]
    for indx in range(len(source_var)):
        ema_op.append(target_var[indx].assign(target_var[indx]*beta + source_var[indx]*(1.-beta)))
    return ema_op  

def copy_params(target_var,source_var):
    copy_op = []
    for indx in range(len(source_var)):
        copy_op.append(target_var[indx].assign(source_var[indx]))
    return copy_op  