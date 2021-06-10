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

num_train = 50 ; 
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
        print('------ Iteration ' + str(i) + ' ------')
        print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True) # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break

def OR_perceptron():
    print("Training OR Peceptron (Gate 1)...")
    training_examples = []
    training_labels = []

    for i in range(100):
        training_examples.append([random.random(), random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 or training_examples[i][1] > 0.75 else 0.0)

    validate_examples = []
    validate_labels = []

    for i in range(100):
        validate_examples.append([random.random(), random.random()])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 or validate_examples[i][1] > 0.75 else 0.0)

    OR = Perceptron(2, bias=-0.5)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        OR.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(OR.weights)
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True) # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 100: 
            break
    

if __name__ == "__main__":
    AND_perceptron()
    OR_perceptron()