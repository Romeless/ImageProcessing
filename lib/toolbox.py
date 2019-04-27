import numpy as np

def normalize(img):
    img = img/img.max() * 255
    return img // 1
    
def pil_to_np(pil):
    return(np.array(pil).astype('int32'))