Hi there, thanks for marking!
If you have any issues, please email at chskea001@myuct.ac.za

When both of the programs are run, I have included some basic output in the terminal

###### Make Commands ######
    'make XOR' - will make the virtual enviroment and install the requirements.txt and then run the XOR.py file.

    'make Classifier' - will make the virtual enviroment and install the requirements.txt and then run the Classifier.py 

    'make' , 'make venv' and 'make install' - will both make the virtual enviroment and install the requirements.txt, but not run the python file 
    'make clean' - will remove the virtual enviroment as well as all .pyc files.

###### Run Commands ######
    if make/make venv/ make install run:
        requirements.txt will be installed and virtual enviroment created but not entered into.
        To enter virtual enviroment:
            source ./venv/bin/activate
    
    Then can run either XOR.py or Classifier.py by the commands listed below.

###### XOR.py ######

Everything about the XOR.py file is explained in detail in the attached report. 

To run -> python XOR.py

###### Classifier ######

## Important ##
Location of MNIST data set:
When run the program will ask you to please enter the path to the MNIST dataset.
Enter the path here. 

Warning: If you enter the incorrect path or the path does not exist then it will begin
    to download the data. I could easily have changed this by setting download = False,
    however I wanted to ensure that you would be able to mark my working code, so would rather include the option to download the dataset
    if was needed. 

Everything else about the Classifier.py file is explained in detail in the attached report. 

To run -> python Classifier.py


###### Git ######
There is a git log included of all my commits. 
