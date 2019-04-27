from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import os
import time

import lib.canny as cny
        
class ImageProcessing:
    def open_file(self):
        filetypes = [('Image Files', ('.png', 'jpg')), ('All Filetype', '.*') ]
        self.pathname = filedialog.askopenfilename(parent=self.root, initialdir= os.getcwd(), title="Choose an image", filetypes=filetypes)
        
        if self.pathname != None:
            self.load_image()

    def load_image(self):
        self.image = Image.open(self.pathname)
        
        self.photo = self.pil_resize()
        
        self.canvas = Canvas(self.root, width=self.im_w + 1, height= self.im_h + 1)
        self.imgArea = self.canvas.create_image(self.im_w / 2, self.im_h / 2, image = self.photo)
        self.canvas.grid(row=0, column=0, sticky=N+W)
        
        #Label(self.image_frame, text="H: {} ; W: {}".format(self.im_w, self.im_h)).grid(row=1)
        
    def refresh_image(self, im):
        self.image = im
        self.photo = self.pil_resize()
        
        self.canvas.itemconfig(self.imgArea, image = self.photo)
        #Label(self.root, text="H: {} ; W: {}".format(self.im_w, self.im_h)).grid(row=1)
    
    def undo(self):
        self.refresh_image(self.last_image.copy())
        
    def gaussian_image_segmentation(self):
        pass
           
    def canny_window(self):
        canny_window = Toplevel(self.root)
        canny_window.title("Canny Edge Finding")
        
        Label(canny_window, text="Window").grid(row=0, sticky=W)
        Label(canny_window, text="Weight").grid(row=1, sticky=W)
        
        canny_entry_window = Entry(canny_window)
        canny_entry_weight = Entry(canny_window)
        canny_entry_window.insert(0,"3")
        canny_entry_weight.insert(0,"2")
        canny_entry_window.grid(row=0, column=1)
        canny_entry_weight.grid(row=1, column=1)
        
        self.canny_button = Button(canny_window, text="SUBMIT", 
                            command= lambda: self.canny_edge_finding(
                                            int(canny_entry_window.get()),
                                            float(canny_entry_weight.get())
                                            )
                                    )
        self.canny_button.grid(row=0, column=2, rowspan=2, sticky=N+S+E+W)
    
        
    def canny_edge_finding(self, window, weight):
        
        self.last_image = self.image.copy()
        res = cny.canny(im = self.image, w = window, weight = weight)
        res = Image.fromarray(cny.normalize(res).astype('int8'))
        
        self.refresh_image(res)
        
    def save_image(self):
        desired_extension = '.jpg'
        basename = os.path.basename(self.pathname)
        imagename_no_ext = basename[:basename.rindex('.')]
        
        self.image.convert("RGB").save("{}{}{}".format("outputImage/", imagename_no_ext, desired_extension))
        
    def __init__(self, master):
        self.root = master
        
        self.prepare_menubar()
        
    def prepare_menubar(self):
        main_menu = Menu(self.root)
        
        filemenu = Menu(main_menu, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_file)
        filemenu.add_command(label="Save", command=self.save_image)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=root.quit)
        main_menu.add_cascade(label="File", menu=filemenu)
        
        editmenu = Menu(main_menu, tearoff=0)
        editmenu.add_command(label="Undo", command=self.undo)
        main_menu.add_cascade(label="Edit", menu=editmenu)
        
        adveditmenu = Menu(main_menu, tearoff=0)
        adveditmenu.add_command(label="K-Means Segm", command=root.quit)
        adveditmenu.add_command(label="Canny Edge", command=self.canny_window)
        main_menu.add_cascade(label="Advanced", menu=adveditmenu)
        
        helpmenu = Menu(main_menu, tearoff=0)
        helpmenu.add_command(label="???", command=root.quit)
        helpmenu.add_command(label="About", command=root.quit)
        main_menu.add_cascade(label="Help", menu=helpmenu)
        
        self.root.config(menu=main_menu)
        
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

root.mainloop()