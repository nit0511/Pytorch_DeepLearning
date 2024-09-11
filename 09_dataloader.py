''' 
epoch = 1 forward and backward pass of All training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iteration for 1 epochs
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading

        xy = np.loadtxt('D:/DeepLeraningProject/Autoencoder/image-deblurring-using-deep-learning/Pytorch_tutorial/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.X = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True,num_workers=0)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epochs in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward , update
        if (i + 1) % 5 == 0:
            print(f"epoch {epochs +1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}")   


# torchvision.datasets.MNIST()
#fashion-mnist, cifar, coco