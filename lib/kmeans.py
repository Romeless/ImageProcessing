import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
    
def kmeans(img, k, tol = 0.001, max_iter = 300):
    img = pil_to_np(img)
    
    shape = img.shape
    img = img.reshape((-1,3))
    img = np.float32(img)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    comp, label, centers = cv.kmeans(img, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    res = res.reshape((shape))
    return Image.fromarray(normalize(res).astype('uint8'))
    
if __name__ == '__main__':
    import sys
    from toolbox import *
    
    if len(sys.argv) >= 2:
        img = sys.argv[1]
        k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        tol = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.001
        max_iter = int(sys.argv[4]) if len(sys.argv) >= 5 else 300
        
        res = kmeans(img, k=k, tol=tol, max_iter=max_iter)
        
        save_image(res, sys.argv[1])
else:
    from lib.toolbox import *