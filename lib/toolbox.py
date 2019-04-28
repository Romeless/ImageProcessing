import numpy as np
from PIL import Image
import time
import os
from numpy.lib.stride_tricks import as_strided

def normalize(img):
    img = img/img.max() * 255
    return img // 1
    
def pil_to_np(pil):
    return(np.array(pil).astype('int32'))
    
def grayscale(im):
    return np.dot(im, [0.3, 0.587, 0.114])
    
def prepare_gaussian_kernel(val=1.4, w=3):
    t = np.linspace(-1 * val, val, w)
    bump = np.exp(-0.1 * t ** 2)
    return bump / np.trapz(bump)

def gaussian_kernel_2d(G):
    return G[:,np.newaxis] * G[np.newaxis,:]
    
def get_all_window(M, w):
    M = np.pad(M, w//2, 'symmetric')
    sub_shape = (w, w)
    view_shape = tuple(np.subtract(M.shape, sub_shape) + 1) + sub_shape
    arr_view = as_strided(M, view_shape, M.strides * 2)
    arr_view = arr_view.reshape((-1,) + sub_shape)
    return arr_view
    
def sobel_op():
    return (np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

def scharr_op():
    return (np.array([[-47,0,47],[-162,0,162],[-47,0,47]]), np.array([[47,162,47],[0,0,0],[-47,-162,-47]]))
    
def save_image(img, filename, desired_extension=".jpg"):
    basename = os.path.basename(filename)
    imagename_no_ext = basename[:basename.rindex('.')]
        
    img.convert("RGB").save("{}{}-{}{}".format("outputImage/", imagename_no_ext, time.time(), desired_extension))