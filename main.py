from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import os
import time
import lib.canny as cny
import lib.kmeans as km
import lib.segmentation as ms
from lib.toolbox import *

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
        
        self.im_w, self.im_h = self.image.size
        Label(self.root, text="H: {}, W: {}".format(self.im_w, self.im_h)).grid(row=1, column=0, sticky=E)
    
    def save_image(self):
        save_image(self.image, self.pathname)
        
    def refresh_image(self, im):
        self.image = im
        self.photo = self.pil_resize()
        
        self.canvas.itemconfig(self.imgArea, image = self.photo)
        
        self.im_w, self.im_h = self.image.size
        Label(self.root, text="H: {}, W: {}".format(self.im_w, self.im_h)).grid(row=1, column=0, sticky=E)
      
    def undo(self):
        self.refresh_image(self.last_image.copy())
    
    def resize_window(self):
        resize_window = Toplevel(self.root)
        resize_window.title("Resize")
        
        Label(resize_window, text="Ratio").grid(row=1, sticky=W)
        
        resize_ratio_entry = Entry(resize_window)
        resize_ratio_entry.insert(0, "1.0")
        resize_ratio_entry.grid(row=1, column=1, sticky=W)
        
        resize_button = Button(resize_window, text="Resize",
                                command= lambda: self.resize(
                                float(resize_ratio_entry.get())
                                )
                               )
        resize_button.grid(row=0, column = 2, rowspan=2, sticky= N+E+W+S)
        
    def resize(self, ratio):
        self.im_w, self.im_h = self.image.size
        
        self.image = self.image.resize((int(self.im_w * ratio), int(self.im_h * ratio)), Image.NEAREST)     
        self.refresh_image(self.image)
        print("resized")
        
    def kmeans_window(self):
        kmeans_window = Toplevel(self.root)
        kmeans_window.title("K-Means Segmentation")
        
        Label(kmeans_window, text="Cluster").grid(row=0, sticky=W)
        Label(kmeans_window, text="Tolerance").grid(row=1, sticky=W)
        Label(kmeans_window, text="Max Iterations").grid(row=2, sticky=W)
        
        kmeans_entry_cluster = Entry(kmeans_window)
        kmeans_entry_tol = Entry(kmeans_window)
        kmeans_entry_iter = Entry(kmeans_window)
        kmeans_entry_cluster.insert(0,"5")
        kmeans_entry_tol.insert(0,"0.0001")
        kmeans_entry_iter.insert(0,"300")
        kmeans_entry_cluster.grid(row=0, column=1)
        kmeans_entry_tol.grid(row=1, column=1)
        kmeans_entry_iter.grid(row=2, column=1)
        
        self.kmeans_button = Button(kmeans_window, text="SUBMIT", 
                            command= lambda: self.kmeans(
                                            int(kmeans_entry_cluster.get()),
                                            float(kmeans_entry_tol.get()),
                                            int(kmeans_entry_iter.get())
                                            )
                                    )
        self.kmeans_button.grid(row=0, column=2, rowspan=2, sticky=N+S+E+W)
        
        self.kmeans_w_undo = Button(kmeans_window, text="UNDO", command=self.undo)
        self.kmeans_w_undo.grid(row=2, column=2)
    
    def kmeans(self, cluster, tol, iter):
        self.last_image = self.image.copy()
        
        res = km.kmeans(self.image, k = cluster, tol = tol, max_iter = iter)
        
        self.refresh_image(res)     
        
    def meanshift_window(self):
        meanshift_window = Toplevel(self.root)
        meanshift_window.title("Mean Shift Segmentation")
        
        Label(meanshift_window, text="Spatial Radius").grid(row=0, sticky=W)
        Label(meanshift_window, text="Range Radius").grid(row=1, sticky=W)
        Label(meanshift_window, text="Min Density").grid(row=2, sticky=W)
        Label(meanshift_window, text="Speedup Level (0,1,2)").grid(row=3, sticky=W)
        Label(meanshift_window, text="This will possibly take a lot of time").grid(row=4, sticky=W)
        
        meanshift_entry_spatial = Entry(meanshift_window)
        meanshift_entry_range = Entry(meanshift_window)
        meanshift_entry_dens = Entry(meanshift_window)
        meanshift_entry_speedup = Entry(meanshift_window)
        meanshift_entry_spatial.insert(0,"3")
        meanshift_entry_range.insert(0,"1.4")
        meanshift_entry_dens.insert(0,"50")
        meanshift_entry_speedup.insert(0,"2")
        meanshift_entry_spatial.grid(row=0, column=1)
        meanshift_entry_range.grid(row=1, column=1)
        meanshift_entry_dens.grid(row=2, column=1)
        meanshift_entry_speedup.grid(row=3, column=1)
        
        self.meanshift_button = Button(meanshift_window, text="SUBMIT", 
                                        command= lambda: self.meanshift(
                                            int(meanshift_entry_spatial.get()),
                                            float(meanshift_entry_range.get()),
                                            int(meanshift_entry_dens.get()),
                                            int(meanshift_entry_speedup.get())
                                            )
                                    )
        self.meanshift_button.grid(row=0, column=2, rowspan=2, sticky=N+S+E+W)
        
        
        self.meanshift_w_undo = Button(meanshift_window, text="UNDO", command=self.undo)
        self.meanshift_w_undo.grid(row=4, column=2, sticky=S+E)
        
    def meanshift(self, spatial, range, density, speed):
        self.last_image = self.image.copy()
        
        segm = ms.Segmenter(spatial_radius=spatial, range_radius=range,
                        min_density = density, speedup_level=speed)
        (res, labels, nb_regions) = segm.segmentate(normalize(np.array(self.image)).astype('uint8'))
        res = Image.fromarray(normalize(res).astype('uint8'))
        
        self.refresh_image(res)
        
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
        self.canny_w_undo = Button(canny_window, text="UNDO", command=self.undo)
        self.canny_w_undo.grid(row=2, column=2)
    
    def canny_edge_finding(self, window, weight):
        
        self.last_image = self.image.copy()
        res = cny.canny(im = self.image, w = window, weight = weight)
        res = Image.fromarray(normalize(res).astype('uint8'))
        
        self.refresh_image(res)
        
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
        editmenu.add_command(label="Resize", command=self.resize_window)
        main_menu.add_cascade(label="Edit", menu=editmenu)
        
        adveditmenu = Menu(main_menu, tearoff=0)
        adveditmenu.add_command(label="K-Means Segm", command=self.kmeans_window)
        adveditmenu.add_command(label="Mean-Shift Segm", command=self.meanshift_window)
        adveditmenu.add_command(label="Canny Edge", command=self.canny_window)
        main_menu.add_cascade(label="Advanced", menu=adveditmenu)
        
        helpmenu = Menu(main_menu, tearoff=0)
        helpmenu.add_command(label="HELP", command=root.quit)
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