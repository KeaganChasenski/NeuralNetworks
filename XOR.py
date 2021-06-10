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

def generate_training_set():
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.8 and training_examples[i][1] > 0.8 else 0.0)
    
    return training_examples, training_labels 

def generate_validation_set():

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.8 and validate_examples[i][1] > 0.8 else 0.0)

    return validate_examples, validate_labels

def AND_perceptron():
    training_examples, training_labels = generate_training_set()
    validate_examples, validate_labels = generate_validation_set()

    #print(training_examples)
    #print(training_labels)

    AND = Perceptron(2, bias=-1.0)
    #print(AND.weights)
    valid_percentage = 0
    #print(valid_percentage)ß

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

if __name__ == "__main__":
    AND_perceptron()