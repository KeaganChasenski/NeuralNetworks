# Assignment 3 - CNNS
    #XOR.py

# Keagan Chasenski
# CHSKEA001
# 10 June 2021

#Returns
    #

import numpy as np
import random
from  Perceptron import Perceptron
import sys 

num_train = 500 ; 
learning_rate = 0.2 ; 

def AND_perceptron():
    print("Training AND Peceptron (Gate 0)...")
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)

    AND = Perceptron(2, bias=-1.5)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        AND.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break
    print("Accuracy = ", valid_percentage)
    return AND

def OR_perceptron():
    print("Training OR Peceptron (Gate 1)...")
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 or training_examples[i][1] > 0.75 else 0.0)

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 or validate_examples[i][1] > 0.75 else 0.0)

    OR = Perceptron(2, bias=-0.75)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        OR.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(OR.weights)
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 100: 
            break
    print("Accuracy = ", valid_percentage)
    return OR

def NAND_perceptron():
    print("Training NAND Peceptron (Gate 2)...")
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 1)

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels.append(0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 1)

    NAND = Perceptron(2, bias=0.75)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        NAND.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(NAND.weights)
        valid_percentage = NAND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break
    print("Accuracy = ", valid_percentage)
    return NAND

def XOR_perceptron(x1,x2):
    x = [x1, x2]

    xor = AND.predict([OR.predict(x), NAND.predict(x)])

    if xor is True:
        return 1
    if xor is False:
        return 0

if __name__ == "__main__":
    AND = AND_perceptron()
    OR = OR_perceptron()
    NAND = NAND_perceptron()
    
    print("Please enter two inputs:")
    x1, x2 = map(float, input().split())
    xor = XOR_perceptron(x1,x2)
    print(xor)
    