import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
from lib.toolbox import *
    
def kmeans(data, k, tol = 0.001, max_iter = 300):
    data = pil_to_np(data)
    
    shape = data.shape
    data = data.reshape((-1,3))
    data = np.float32(data)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    comp, label, centers = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    res = res.reshape((shape))
    return Image.fromarray(normalize(res).astype('uint8'))
    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 2:
        data = sys.argv[1]
        
        k = int(sys.argv[2])
        
        if len(sys.argv) == 5:
            tol = float(sys.argv[3])
            max_iter = int(sys.argv[4])
            
            res = kmeans(data, k=k, tol=tol, max_iter=max_iter)
        else:
            res = kmeans(data, k=k)
        
        desired_ext = '.jpg'
        filename = sys.argv[1][:sys.argv[1].rindex('.')]
        
        res.save("{}-out{}".format(filename ,desired_ext))