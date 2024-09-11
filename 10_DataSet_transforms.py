'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data during creation of the DataSet

complete list of built-in transforms:
https://pytoch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandoCrop, RAndomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndarray
ToTensor: from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),RandomCrop(224)])

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading

        xy = np.loadtxt('D:/DeepLeraningProject/Autoencoder/image-deblurring-using-deep-learning/Pytorch_tutorial/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here
        self.X = xy[:,1:]
        self.y = xy[:,[0]] # n_samples, 1

        self.transform = transform
        

    def __getitem__(self, index):
        sample =  self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

            return sample
        return sample
    
    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

    
    

    
    
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])