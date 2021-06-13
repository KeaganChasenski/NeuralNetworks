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

    return images, labels

def plot_data(images, labels):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def build_Model():
    model=nn.Sequential(nn.Linear(784,128), # 1 layer:- 784 input 128 o/p
            nn.ReLU(),          # Defining Regular linear unit as activation
            nn.Linear(128,64),  # 2 Layer:- 128 Input and 64 O/p
            nn.Tanh(),          # Defining Regular linear unit as activation
            nn.Linear(64,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)
            nn.LogSoftmax(dim=1) # Defining the log softmax to find the probablities for the last output unit
            ) 

    print(model)
if __name__ == "__main__":
    images, labels = load_data()
    plot_data(images, labels)
    build_Model()