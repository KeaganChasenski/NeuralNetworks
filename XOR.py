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
num_train = 1000 ; 

def AND_perceptron():
    print("Training AND Peceptron (Gate 0)...")

    # Blank training examples and labels array
    training_examples = []
    training_labels = []

    # Randomly choose a number beteween -0.25, 1.25 for each input [x1,x2] of the index of the array
    # Since we want the perceptron to be noise positive, the label works on a 1 if it is greater than 0.75
    for i in range(num_train):
        training_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        # For an and gate, both the [i][0] AND [i][1] must be postive (>0.75) to recieve a label as 1
        training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)

    # Create Blank Validation examples
    validate_examples = []
    validate_labels = []
    
    # The loop follows the same logic as the training examples being produced above
    # Except these values are kept exclusively for validating a trained perceptron
    for i in range(num_train):
        validate_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)

    # Create the AND perceptron from the Perceptron.py class
    # 2 inputs, and bias value = -1.5
    AND = Perceptron(2, bias=-1)
    valid_percentage = 0

    # Loop to continuely train the modle until we reach an accuracy of 98%
    # Loop counter
    i = 0
    while valid_percentage < 0.9: 
        i += 1

        # Train AND perceptron using training examples and labels
        # Learning rate for AND = 0.2
        AND.train(training_examples, training_labels, 0.2) 

        # Once we have trained, validate the modle
        # returns the number of correct predictions / total predictions
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        #if i == num_train: 
        #    break

    # Display the accuracy of this perceptron, and how many iterations it took of training to achieve this
    print("Accuracy = ", valid_percentage, "in", i , "iterations")

    # Return the model
    return AND #, valid_percentage

def OR_perceptron():
    print("Training OR Peceptron (Gate 1)...")

    # Blank training examples and labels array
    training_examples = []
    training_labels = []

    # Randomly choose a number beteween -0.25, 1.25 for each input [x1,x2] of the index of the array
    # Since we want the perceptron to be noise positive, the label works on a 1 if it is greater than 0.75
    for i in range(num_train):
        training_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        # For an OR gate, either the [i][0] or [i][1] must be postive (>0.75) to recieve a label as 1
        training_labels.append(1.0 if training_examples[i][0] > 0.75 or training_examples[i][1] > 0.75 else 0.0)

    # Create Blank Validation examples
    validate_examples = []
    validate_labels = []

    # The loop follows the same logic as the training examples being produced above
    # Except these values are kept exclusively for validating a trained perceptron
    for i in range(num_train):
        validate_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 or validate_examples[i][1] > 0.75 else 0.0)

    # Create the OR perceptron from the Perceptron.py class
    # 2 inputs, and bias value = -1
    OR = Perceptron(2, bias=-1)
    valid_percentage = 0

    # Loop to continuely train the modle until we reach an accuracy of 98%
    # Loop counter
    i = 0
    while valid_percentage < 0.8:
        i += 1

        # Train OR perceptron using training examples and labels
        # Learning rate for OR = 0.4
        OR.train(training_examples, training_labels, 0.4)  
        
        # Once we have trained, validate the modle
        # returns the number of correct predictions / total predictions
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=False) 

    # Display the accuracy of this perceptron, and how many iterations it took of training to achieve this    
    print("Accuracy = ", valid_percentage, "in ", i , "i terations")

    # Return the model
    return OR

def NOT_perceptron():
    print("Training NOT Peceptron (Gate 3)...")

    # Blank training examples and labels array
    training_examples = []
    training_labels = []

    # Randomly choose a number beteween -0.25, 1.25 for the one input [x1] at the index of the array
    # Since we want the perceptron to be noise positive, the label works on a 0 if it is greater than 0.75
    for i in range(num_train):
        training_examples.append([random.uniform(-0.25,1.25)])
        # For an NOT gate, if the input is <0.75 then the output is 1
        # and if the input is >0.75, then the output is 0
        training_labels.append(0 if training_examples[i][0] > 0.75  else 1)

    # Create Blank Validation examples
    validate_examples = []
    validate_labels = []

    # The loop follows the same logic as the training examples being produced above
    # Except these values are kept exclusively for validating a trained perceptron
    for i in range(num_train):
        validate_examples.append([random.uniform(-0.25,1.25)])
        validate_labels.append(0 if validate_examples[i][0] > 0.75  else 1)

    # Create the NOT perceptron from the Perceptron.py class
    # 1 inputs, and bias value = 0.5
    NOT = Perceptron(1, bias=0.50)
    valid_percentage = 0

    # Loop to continuely train the modle until we reach an accuracy of 98%
    # Loop counter
    i = 0
    while valid_percentage < 0.98: 
        i += 1

        # Train NOT perceptron using training examples and labels
        # Learning rate for NOT = 0.4
        NOT.train(training_examples, training_labels, 0.2)  

        # Once we have trained, validate the modle
        # returns the number of correct predictions / total predictions
        valid_percentage = NOT.validate(validate_examples, validate_labels, verbose=False) 

    # Display the accuracy of this perceptron, and how many iterations it took of training to achieve this
    print("Accuracy = ", valid_percentage, "in", i , "iterations")

    # Return the model
    return NOT

def XOR_perceptron(x1,x2):
    # Create array from two x inputs
    x = [x1, x2]

    # Input of NOT gate from the output of AND gate
    not_gate = NOT.predict(AND.predict(x))

    # XOR is the ouput of the Not gate (from an AND gate) ANDED with the output from a OR gate
    xor = AND.predict([OR.predict(x), not_gate])

    if xor is True:
        return 1
    if xor is False:
        return 0

if __name__ == "__main__":
    # Construct each perceptron
    #AND_accuracy = 0
    #while AND_accuracy < 0.98:

    #AND, AND_accuracy = AND_perceptron()
    AND = AND_perceptron()

    OR = OR_perceptron()

    NOT = NOT_perceptron()
    
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