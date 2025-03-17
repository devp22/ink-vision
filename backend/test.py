import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# trainset = datasets.MNIST(root='./data',train=True,download=True)

'''Creating transformer to normalize data'''
# The MNIST dataset has pixel values in the range [0, 255]. Normalization scales them to [0, 1] (via ToTensor()) and further to zero-centered values using (mean=0.1307, std=0.3081), which improves model training.

# print(f"trainset mean: {trainset.data.float().mean()/255}") # 0.1306
# print(f"trainset std: {trainset.data.float().std()/255}") # 0.308

transFormer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # Normalization is done with mean (0.1307) and std (0.308)
    ])
print("Transformer created")
'''Loading the data set'''
# Dataset is loaded from MNIST library and is already splitted in training and testinf

train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transFormer)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transFormer)
print("Datasets created")

# print(f"dataset has been loaded")
# print(f"length of training dataset is {len(train_dataset)}")
# print(f"length of testing dataset is {len(test_dataset)}")

''' Creating DataLoaders to efficiently batch and shuffle '''
# The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
# You need DataLoader to create mini-batches and shuffle the data for training efficiency.

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2,drop_last=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True,num_workers=2,drop_last=True)
print("DataLoaders created")

