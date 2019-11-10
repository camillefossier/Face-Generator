import os

from PIL import Image, ImageFilter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# TODO : Gather face data

SIZE = 64,64
NB_FEATURES = 50

def handle_image(input_path, output_path):
    im = Image.open(input_path)
    im.thumbnail(SIZE)
    im = im.convert(mode="L")
    im.save(output_path)

def create_dataset(input_folder, output_folder):
    images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for i in images:
        handle_image(os.path.join(input_folder, i), os.path.join(output_folder, i))

def full_dataset(root_folder, output_folder, max_folder=None):
    folders = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    #create_dataset('./PINS/PINS/pins_Aaron Paul', './small_faces')
    if max_folder is not None:
        folders = folders[:max_folder]

    for f in folders:
        create_dataset(f, output_folder)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(NB_FEATURES, 215)
        self.lin2 = nn.Linear(215, 940)
        self.lin3 = nn.Linear(940, SIZE[0] * SIZE[1])

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x

class DataSet:

    def __init__(self, features, images):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """

        self.features = features
        self.images = images

    def __len__(self):
        """return number of points in our dataset"""
        return self.features.shape[0]

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """

        feature = self.features[idx,:]
        image = self.images[idx,:]

        return feature, image

if __name__ == "__main__":
    
    # ONLY TO REGENERATE : a bit long
    #full_dataset('./PINS/PINS', './small_faces')

    # TODO : Load data as matrix

    folder = './small_faces'
    n = 5000
    p = SIZE[0] * SIZE[1]
    pics = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    data = np.zeros((n, p))
    for i in range(n):
        im = Image.open(os.path.join(folder, pics[i])).convert("L")
        v = np.array(im).flatten()
        data[i,:len(v)] = v
    print(data)

    # PCA on data

    pca = PCA(n_components=NB_FEATURES)
    pca.fit(data)
    #plt.plot(pca.explained_variance_ratio_)
    #plt.show()

    X = pca.transform(data)
    X = np.apply_along_axis(lambda c: (c - min(c)) / (max(c) - min(c)), 1, X)
    
    # TODO : Establish network with :
    #   n entries
    #   stuff in between
    #   32*32 outputs channels
    PATH = "./face_model.pth"
    net = Net()
    ds = DataSet(X, data/255)

    data_loader = DataLoader(ds, batch_size=10, shuffle=True)
    if (os.path.isfile(PATH)):
        net.load_state_dict(torch.load(PATH))
    else:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(3):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
        torch.save(net.state_dict(), PATH)
        print('Finished Training')
    
    ran = np.random.random_sample((500,NB_FEATURES))
    empty = np.zeros((500, SIZE[0] * SIZE[1]))
    ds_random = DataSet(ran, empty)
    data_loader_random = DataLoader(ds_random, batch_size=1)

    for d in data_loader_random:
        f,_ = d
        out = net(f.float())
        for i, o in enumerate(out):
            plt.imshow((o * 255).view(SIZE).detach().numpy(), cmap="gray")
            plt.show()
        
    for d in data_loader:
        f,l = d
        out = net(f.float())
        for i, o in enumerate(out):
            plt.subplot(121)
            plt.imshow((l[i] * 255).view(SIZE).detach().numpy(), cmap="gray")
            plt.subplot(122)
            plt.imshow((o * 255).view(SIZE).detach().numpy(), cmap="gray")
            plt.show()
    #input_vector = np.random.random_sample(20)


    # TODO : Print image

    # ----------------------------------------------------- #

    # TODO : Try the same thing with a GAN

    print("done")