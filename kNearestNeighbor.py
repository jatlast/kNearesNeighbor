########################################################################
# Jason Baumbach
#   CSC 546 - Homework 7 (due: March 26, 2019 @ 5:59 PM)
#   2.  k-nearest neighbor (5 points) – Write a k-NN classifier for the data 
#       in the files provided. Make your training data from the training data
#       files provided and test on the test file.
#
#       a.  Test it with k=1
#       b.  Test it with k=3
# 
#   For question 2:
#   -   Grads use 2-dimensional data in homework_classify_train_2D.dat and
#       homework_classify_test_2D.dat.
#
# Note: this code is available on GitHub 
#   https://github.com/jatlast/kNearestNeighbor.git
#
# The following websites were referenced:
#   For Python scatterplot tools
#   https://pythonspot.com/matplotlib-scatterplot/
#
########################################################################

# required for sqrt function in Euclidean Distance calculation
import math
# required for scatterplot visualization
import numpy as np
import matplotlib.pyplot as plt
# required for parsing data files
import re

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform k-nearest neighbor classification on 2-dimensional data provided. (Note: 2D file includes 3 columns: x, y, classification")
#parser.add_argument("-f", "--filename", default="./data/homework_classify_train_2D.dat", help="file name (and path if not in . dir)")
parser.add_argument("-n", "--neighbors", type=int, choices=[1, 3], default=1, help="number of nearest neighbors to use")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

# create a dictionary of list objects equal to the number of neighbors to test
neighbors_dict = {}
for i in range(1, args.neighbors + 1):
    neighbors_dict[i] = {'index' : -1, 'distance' : 1000}

if args.verbosity > 0:
    print(f"neighbors={args.neighbors} : len(neighbors_dict)={len(neighbors_dict)}")
#    print(f"filename={args.filename} : neighbors={args.neighbors} : len(neighbors_dict)={len(neighbors_dict)}")

# compute Euclidean distance between any two given vectors with any length.
# Note: adapted from CSC 587 Adv Data Mining, HW02
# Note: a return value < 0 = Error
def EuclideanDistanceBetweenTwoVectors(vOne, vTwo):
    distance = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        return -1

    for p in range(0, v_one_len):
        distance += math.pow((abs(vOne[p] - vTwo[p])), 2)
    return math.sqrt(distance)

# 2D scatterplot
def Scatterplot2D(dTrainingData, vTestData, dNeighbors, dVariables):
    # plot training data points
    for i in range(0, len(dTrainingData)):
        if dTrainingData[i][2] == 1:
            plt.plot(dTrainingData[i][0], dTrainingData[i][1], dVariables['dColors'][1][0])
        else:
            plt.plot(dTrainingData[i][0], dTrainingData[i][1], dVariables['dColors'][2][0])
    # plot test data points
        if dTrainingData[i][2] == 1:
            plt.plot(vTestData[0], vTestData[1], dVariables['dColors'][1][1])
        else:
            plt.plot(vTestData[0], vTestData[1], dVariables['dColors'][2][1])
    plt.title(dVariables['plot_title'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

variables_dict = {
    'training_file' : "./data/homework_classify_train_2D.dat"
    , 'testing_file' : "./data/homework_classify_test_2D.dat"
    , 'plot_title' : 'Default Title'
    , 'dColors' : { # plotting shapes and colors
        1 : ['k.', 'bo']    # black . data and blue o mean
        , 2 : ['g+', 'rP']  # green + data and red "filled +" mean
        , 3 : ['cx', 'mX']  # cyan x data and magenta "filled x" mean
    }
}

# Read the provided data files
# Note: These files are not in a generic (',' or '\t') delimited format -- they require parsing
def ReadFileDataIntoDictOfLists(sFileName, dTraingDict):
    # read in the training data file
    with open(sFileName, mode='r') as data_file:
        line_number = 0
        # parse data file
        for line in data_file:
            # a regular expression to match both the 1-dimensional & 2-dimensional files supplied
            match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d)', line)
            # get the x (and possibly the y) values
            if match:
                if match.group(1) and match.group(2) and match.group(3):
                    dTraingDict[line_number] = [float(match.group(1)), float(match.group(2)), int(match.group(3))]
                else:
                    print(f"Warning: all three groups were not found on line ({line})")
            else:
                print(f"Warning: no match for line ({line})")
            line_number += 1

# Populate the k-nearest neighbors by comparing all training data with test data point
def PopulateNearestNeighborsDicOfIndexes(dNeighbors, dTrainingData, vTestData):
    for td in range(0, len(dTrainingData)):
        index_taken = False
        for nn in range(1, len(dNeighbors) + 1):
            temp_vec = (dTrainingData[td][0], dTrainingData[td][1])
            EuclideanDistance = EuclideanDistanceBetweenTwoVectors(vTestData, temp_vec)
            if EuclideanDistance < dNeighbors[nn]['distance'] and not index_taken:
                for nn2 in range(1, len(dNeighbors) + 1):
                    if dNeighbors[nn2]['index'] == td:
                        index_taken = True
                        break
                    else:
                        dNeighbors[nn]['distance'] = EuclideanDistance
                        dNeighbors[nn]['index'] = td

# Load the training data
training_dict = {}
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# Load the testing data
testing_dict = {}
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# Print some of the input file data
if args.verbosity > 1:
    print("The first 5 training samples:")
    for i in range(0, len(training_dict)):
        if i > 4:
            break
        else:
            print(f"\t{i} {training_dict[i]}")
    print("The first 2 testing samples:")
    for i in range(0, len(testing_dict)):
        if i > 1:
            break
        else:
            print(f"\t{i} {testing_dict[i]}")

test_vect = (testing_dict[1][0], testing_dict[1][1])
# create the plot graph
variables_dict['plot_title'] = "Test variable is type {}".format(testing_dict[1][2])
#Scatterplot2D(training_dict, test_vect, neighbors_dict, variables_dict)

PopulateNearestNeighborsDicOfIndexes(neighbors_dict, training_dict, test_vect)

print(f"neighbors_dict: {neighbors_dict}")
