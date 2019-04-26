from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import lib.canny as cny
from pathlib import Path

class ImageProcessing:
    def load_image(self, im=None):
        if self.canvas != None:
            self.canvas.delete("all")
        if im==None:
            im = self.entry_image.get()
        self.image = Image.open(im)
        
        self.photo = self.pil_resize()
        
        self.canvas = Canvas(self.image_frame, width=self.im_w + 1, height= self.im_h + 1)
        self.imgArea = self.canvas.create_image(self.im_w / 2, self.im_h / 2, image = self.photo)
        self.canvas.grid(row=0, column=0, sticky=W)
        
        #Label(self.image_frame, text="H: {} ; W: {}".format(self.im_w, self.im_h)).grid(row=1)
        
    def refresh_image(self, im):
        self.image = im
        self.photo = self.pil_resize()
        
        self.canvas.itemconfig(self.imgArea, image = self.photo)
        #Label(self.root, text="H: {} ; W: {}".format(self.im_w, self.im_h)).grid(row=1)
    
    def canny_edge_finding(self):
        window = int(self.entry_window.get())
        weight = float(self.entry_weight.get())
        
        res = cny.canny(im = self.image, w = window, weight = weight)
        res = Image.fromarray(cny.normalize(res).astype('int8'))
        
        self.refresh_image(res)
        
    def save_image(self):
        pass
                 
    def __init__(self, master):
        self.root = master
        
        self.control_frame = Frame(self.root)
        self.control_frame.grid(row=0, column=0)
        
        self.segmentation_frame = Frame(self.root)
        self.segmentation_frame.grid(row=1, column=0)
        
        self.canny_frame = Frame(self.root)
        self.canny_frame.grid(row=2, column=0)
        
        self.image_frame = Frame(self.root)
        self.image_frame.grid(row=0, column=1, rowspan=3)
        
        self.prepare_image()
        self.prepare_canny()
        
    def prepare_image(self):
        Label(self.control_frame, text="Image Path").grid(row=0, sticky=N+W)
        self.entry_image = Entry(self.control_frame)
        self.entry_image.grid(row=0, column=1, sticky=N+W)
        self.entry_image.insert(0, "inputImage/input.png")
        
        self.canvas = None
        self.load_button = Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=1, column=0, sticky=N+W)
        self.show_button = Button(self.control_frame, text="Refresh Image", command=self.refresh_image)
        self.show_button.grid(row=1, column=1, sticky=N+W)
        self.save_button = Button(self.control_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=2, column=0, sticky=N+W)
        
    def prepare_canny(self):
        Label(self.canny_frame, text="Canny").grid(row=2, sticky=W)
        Label(self.canny_frame, text="Window").grid(row=3, sticky=W)
        Label(self.canny_frame, text="Weight").grid(row=4, sticky=W)
        self.entry_window = Entry(self.canny_frame)
        self.entry_weight = Entry(self.canny_frame)
        self.entry_window.insert(0,"3")
        self.entry_weight.insert(0,"2")
        self.entry_window.grid(row=3, column=1)
        self.entry_weight.grid(row=4, column=1)
        
        self.canny_button = Button(self.canny_frame, text="CANNY", command=self.canny_edge_finding)
        self.canny_button.grid(row=4, column=2, rowspan=2)
    
    def pil_resize(self):
        self.im_w, self.im_h = self.image.size
        if self.im_w > 500 or self.im_h > 500:
            denum1 = 500 / self.im_w
            denum2 = 500 / self.im_h
            if denum1 > denum2:
                self.im_w = self.im_w * denum2
                self.im_h = self.im_h * denum2
            else:
                self.im_w = self.im_w * denum1
                self.im_h = self.im_h * denum1
        
        self.im_w, self.im_h = int(self.im_w), int(self.im_h)
        new_im = self.image.copy()
        new_im = new_im.resize((self.im_w, self.im_h), Image.NEAREST)
        return(ImageTk.PhotoImage(new_im))

root = Tk()

app = ImageProcessing(root)

root.mainloop() # optional; see description below