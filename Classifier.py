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
from time import time 

batch_size_train = 64
batch_size_test = 1000


def load_data():
    print("loading data...")
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

    return images, labels, trainloader

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


    print("building model...")
    model=nn.Sequential(nn.Linear(784,128), # 1 layer:- 784 input 128 o/p
            nn.ReLU(),          # Defining Regular linear unit as activation
            nn.Linear(128,64),  # 2 Layer:- 128 Input and 64 O/p
            nn.Tanh(),          # Defining Regular linear unit as activation
            nn.Linear(64,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)
            nn.LogSoftmax(dim=1) # Defining the log softmax to find the probablities for the last output unit
            ) 

    print(model)

    return model

def loss_function(images, labels, train_loader, model):
    print("defining loss function...")
    # defining the negative log-likelihood loss for calculating loss
    criterion = nn.NLLLoss()
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)

    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL-loss
    
    return loss, criterion

def grad_weights(loss, model):
    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward() # to calculate gradients of parameter 
    print('After backward pass: \n', model[0].weight.grad)

def train(model, train_loader, criterion):
    print("Training neural network...")
    # defining the optimiser with stochastic gradient descent and default parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('Initial weights - ', model[0].weight)

    images, labels = next(iter(train_loader))
    images.resize_(64, 784)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass
    output = model(images)
    loss = criterion(output, labels)
    # the backward pass and update weights
    loss.backward()
    print('Gradient -', model[0].weight.grad)

    time0 = time()
    epochs = 15 # total number of iteration for training
    running_loss_list= []
    epochs_list = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatenning MNIST images with size [64,784]
            images = images.view(images.shape[0], -1) 
        
            # defining gradient in each epoch as 0
            optimizer.zero_grad()
            
            # modeling for each image batch
            output = model(images)
            
            # calculating the loss
            loss = criterion(output, labels)
            
            # This is where the model learns by backpropagating
            loss.backward()
            
            # And optimizes its weights here
            optimizer.step()
            
            # calculating the loss
            running_loss += loss.item()
            
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))


    print("\nTraining Time (in minutes) =",(time()-time0)/60)

def validate(test_loader, model):
    correct_count, all_count = 0, 0
    for images,labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

    
if __name__ == "__main__":

    images, labels, train_loader = load_data()
    plot_data(images, labels)
    model = build_Model()
    loss, criterion = loss_function(images,labels, train_loader, model)
    grad_weights(loss, model)
    train(model, train_loader, criterion)