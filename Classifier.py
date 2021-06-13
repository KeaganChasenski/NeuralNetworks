# Assignment 3 - Artifical Neural Networks
    #Classifier.py

# Keagan Chasenski
# CHSKEA001
# 10 June 2021

#Returns
    # The class (digit) of the image which the user provides the path to (From the MNST10 data set)
    # Creates 
    # 
# Inputs 
    # Path to file the user wants to classify 
    # If want to exit, 'exit' 

import torch 
import torchvision

from torchvision import datasets, transforms
from torch import nn, optim

import matplotlib.pyplot as plt

batch_size_train = 64
batch_size_test = 1000


def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

    testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)

    # creating a iterator
    dataiter = iter(trainloader) 
    # creating images for image and lables for image number (0 to 9) 
    images, labels = dataiter.next() 

    print(images.shape)
    print(labels.shape) 
if __name__ == "__main__":
    load_data()