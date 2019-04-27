import numpy as np
import sys
from lib.toolbox import *

def normalize(img):
    img = img/img.max() * 255
    return img // 1
    
def pil_to_np(pil):
    return(np.array(pil).astype('int32'))

def prepare_gaussian_kernel(val=1.4, w=3):
    t = np.linspace(-1 * val, val, w)
    bump = np.exp(-0.1 * t ** 2)
    return bump / np.trapz(bump)

def gaussian_kernel_2d(G):
    return G[:,np.newaxis] * G[np.newaxis,:]
    
def fft_blurring(im, kernel):
    kernel_ft = np.fft.fft2(kernel, s=im.shape, axes=(0,1)) # turn the kernel to fourier form
    img_ft = np.fft.fft2(im, axes=(0,1))                    # turn the image to fourier form
    img_ft2 = kernel_ft[:,:] * img_ft                       # multiply the fft image and fourier kernel
    img2 = np.fft.ifft2(img_ft2, axes=(0,1)).real           # inverse fft the result of mutiplication
    img2 = img2 / img2.max()
    return img2
    
def fft_gradient(im, op_x, op_y):
    x_ft = np.fft.fft2(op_x, s=im.shape, axes=(0,1))
    y_ft = np.fft.fft2(op_y, s=im.shape, axes=(0,1))
    img_ft = np.fft.fft2(im, axes=(0,1))                    
    img_ft_x = x_ft[:,:] * img_ft
    img_ft_y = y_ft[:,:] * img_ft
    img_x = np.fft.ifft2(img_ft_x, axes=(0,1)).real 
    img_y = np.fft.ifft2(img_ft_y, axes=(0,1)).real 
    
    theta = np.arctan2(np.abs(img_y), img_x)
    theta = theta*180/np.pi
    return (img_x, img_y, theta)
    
def sobel_op():
    return (np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

def scharr_op():
    return (np.array([[-47,0,47],[-162,0,162],[-47,0,47]]), np.array([[47,162,47],[0,0,0],[-47,-162,-47]]))
    
def non_max_sup(G, theta, weight=2, degree=45, maxdeg=180):
    nrows, ncols = G.shape
    res = np.zeros(G.shape)
    ftheta = np.multiply(np.round(G / degree), degree) % maxdeg
    
    for i in range(1, int(nrows) - 1):
        for j in range(1, int(ncols) - 1):
            b1 = 255
            b2 = 255
            
            if (ftheta[i,j] == 0):
                b1 = G[i,j+1]
                b2 = G[i,j-1]
            elif(ftheta[i,j] == 45):
                b1 = G[i+1,j+1]
                b2 = G[i-1,j-1]
            elif(ftheta[i,j] == 90):
                b1 = G[i+1,j]
                b2 = G[i-1,j]
            elif(ftheta[i,j] == 135):
                b1 = G[i,j]
                b2 = G[i-1,j+1]
            
            res[i,j] = G[i,j] * weight + (b1 * -1) + (b2 * -1)
    res[res < 0] = 0
    return res
    
def double_thresholding(img, lowT, highT):
    strong = (img > highT).astype('int32')
    weak = np.bitwise_and(img >= lowT, img <= highT).astype('int32')
    return (strong, weak)
    
def otsu_method(im):
    im = normalize(im).astype('int32')
    total_pixel = im.shape[0] * im.shape[1]
    mean_denom = 1.0 / total_pixel
    hist, bins = np.histogram(im, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(hist[:t])
        pcf = np.sum(hist[t:])
        Wb = pcb * mean_denom
        Wf = pcf * mean_denom

        mub = np.sum(intensity_arr[:t]*hist[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*hist[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    return final_thresh
    
def hysteresis(strong, im_weak):
    nrows, ncols = strong.shape
    res = strong.copy()
    weak = im_weak.copy()
    
    for i in range (1,nrows-1):
        for j in range (1,ncols-1):
            if strong[i,j] != 0:
                #print("Strong -> i = {} : j = {}".format(i,j))
                (res, weak) = spreading(res, weak, i, j)
                
    return res

def spreading(src, weak, loc_i, loc_j):
    for i in range(-1,2):
        for j in range(-1,2):
            if weak[loc_i + i,loc_j + j] != 0:
                #print("Weak -> loc_i = {} : i = {} : loc_j = {} : j = {}".format(loc_i, i, loc_j, j))
                weak[loc_i + i,loc_j + j] = 0
                src[loc_i + i,loc_j + j] = 1
                (src, weak) = spreading(src, weak, loc_i + i, loc_j + j)
    return (src, weak)
    
def canny(im, w=3, weight=2.7):
    im = pil_to_np(im)
    imgrey = np.dot(im, [0.3, 0.587, 0.114])
    
    #Step 1
    kernel = prepare_gaussian_kernel()
    kernel2d = gaussian_kernel_2d(kernel)
    imgrey_blur = fft_blurring(imgrey, kernel2d)
    
    #Step 2
    Kx, Ky = scharr_op()
    Gx, Gy, theta = fft_gradient(imgrey_blur, Kx, Ky)
    im_G = np.abs(Gx) + np.abs(Gy)
    
    #Step 3
    im_nms = non_max_sup(im_G, theta, weight=weight)
    
    #Step 4
    highT = otsu_method(im_nms)
    lowT = highT / 2
    strong, weak = double_thresholding(im_nms, lowT, highT)
    
    #Step 5
    last_im = hysteresis(strong, weak)
    
    return last_im

def save_image(img):
    from PIL import Image
    
    desired_extension = '.jpg'
    imagename_no_ext = sys.argv[1][:sys.argv[1].rindex('.')]
    Image.fromarray(normalize(img).astype('uint8')).save("{}{}".format(imagename_no_ext, desired_extension))
    
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        
        im = plt.imread(path)
        
        if len(sys.argv) == 4:
            w = int(sys.argv[2])
            we = float(sys.argv[3])
            cny = canny(im, w, we)
        else:
            cny = canny(im)
        save_image(cny)
    else:
        print('Error arguments')
        exit()