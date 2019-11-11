import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tkinter as tk
from face_generator import DataSet, Net, SIZE

import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from PIL import Image, ImageTk

class FeaturesInterface(tk.Tk):
    def __init__(self, network, nb_features):
        super(FeaturesInterface, self).__init__()
        self.network = network
        self.nb_features = nb_features
        self.sliders = []
        self.construct_window()

    def update_image(self, event):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        value = []
        for s in self.sliders:
            value.append(s.get())
        ds = DataSet(np.array([value]), np.zeros((1, 64*64)))
        data_loader = DataLoader(ds, batch_size=1)
        for data in data_loader:
            f,_ = data
            out = self.network(f.float())
            for o in out:
                o = (o * 255).view(SIZE).detach().numpy()
                self.im=Image.fromarray(o).resize((600,600))
                self.photo = ImageTk.PhotoImage(image=self.im)
                canvas = tk.Canvas(self.image_frame, width=600, height=600)
                canvas.create_image((300,300), image=self.photo)
                canvas.pack()
                break

    def construct_window(self):
        self.image_frame = tk.Frame(self)
        self.cursor_frame = tk.Frame(self, width=600, height=600)

        for i in range(self.nb_features):
            self.sliders.append(tk.Scale(self.cursor_frame, orient=tk.HORIZONTAL, from_=0, to=1, resolution=0.001, command=self.update_image))
            self.sliders[i].set(0.5)
            #self.sliders[i].bind("<ButtonRelease-1>", self.update_image)
            self.sliders[i].pack()

        self.image_frame.pack(side=tk.LEFT)
        self.cursor_frame.pack(side=tk.RIGHT)

if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load("./face_model.pth"))

    fi = FeaturesInterface(net, 50)
    fi.mainloop()