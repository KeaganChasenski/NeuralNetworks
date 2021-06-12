# Assignment 3 - Artifical Neural Networks
    #XOR.py

# Keagan Chasenski
# CHSKEA001
# 10 June 2021

#Returns
    # The class (0/1) of two inputs passed by user based of XOR logic gate
    # Creates XOR peceptron from output of 3 perceptrons
    # AND, NOT and OR perceptrons
# Inputs 
    # x1 x2 when promted
    # If want to exit, 'exit' 

import random
from  Perceptron import Perceptron

#HyperParamter - defines how many train and validation expamples to create
num_train = 500 ; 

def AND_perceptron():
    print("Training AND Peceptron (Gate 0)...")
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)

    validate_examples = []
    validate_labels = []
    
    for i in range(num_train):
        validate_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)

    
    AND = Perceptron(2, bias=-1.5)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == num_train: 
            break
    print("Accuracy = ", valid_percentage, "in", i , "iterations")
    return AND, valid_percentage

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

    OR = Perceptron(2, bias=-1)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        OR.train(training_examples, training_labels, 0.4)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(OR.weights)
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == num_train: 
            break
    print("Accuracy = ", valid_percentage, "in ", i , "i terations")
    return OR, valid_percentage

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

    NAND = Perceptron(2, bias=0.70)
    valid_percentage = 0

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        NAND.train(training_examples, training_labels, 0.1)  # Train our Perceptron
        #print('------ Iteration ' + str(i) + ' ------')
        #print(NAND.weights)
        valid_percentage = NAND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        #print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == num_train: 
            break
    print("Accuracy = ", valid_percentage, "in ", i , "i terations")
    return NAND

def NOT_perceptron():
    print("Training NOT Peceptron (Gate 3)...")
    training_examples = []
    training_labels = []

    for i in range(num_train):
        training_examples.append(random.random())
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(0 if training_examples[i] > 0.75  else 1)

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append(random.random())
        validate_labels.append(0 if validate_examples[i] > 0.75  else 1)

    NOT = Perceptron(1, bias=0.50)
    valid_percentage = 0

    print(NOT.weights, NOT.num_inputs)
    print(training_examples)
    print(training_examples[0])
    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1

        NOT.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(NAND.weights)
        valid_percentage = NAND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == num_train: 
            break
    print("Accuracy = ", valid_percentage, "in ", i , "i terations")
    return NOT

def XOR_perceptron(x1,x2):
    x = [x1, x2]

    xor = AND.predict([OR.predict(x), NAND.predict(x)])

    if xor is True:
        return 1
    if xor is False:
        return 0

if __name__ == "__main__":
    # Construct each perceptron
    AND_accuracy = 0
    while AND_accuracy < 0.98:
        AND, AND_accuracy = AND_perceptron()

    OR_accuracy = 0 
    while OR_accuracy < 0.98:
        OR, OR_accuracy = OR_perceptron()

    #NAND = NAND_perceptron()

    #NOT = NOT_perceptron()
    
    # Build Network of perceptrons
    print("Constructing Network...")
    print("Done!") 

    # User input
    print("Please enter two inputs:")
    s = input()
    
    # Exit condition for while loop = 'exit' from user
    while (s != 'exit'):
        # split values from user input
        x1, x2 = map(float, s.split())
        # Define XOR predicition based of user input values
        xor = XOR_perceptron(x1,x2)

        # Display prediction, then loop 
        print("XOR Gate: ", xor)
        print("Please enter two inputs:")
        s = input()

    print("Exiting...")