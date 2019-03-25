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
parser = argparse.ArgumentParser(description="Perform k-nearest neighbor classification on 2-dimensional data provided. (Note: 2D files include 3 columns: x, y, class)")
#parser.add_argument("-f", "--filename", default="./data/homework_classify_train_2D.dat", help="file name (and path if not in . dir)")
parser.add_argument("-n", "--neighbors", type=int, choices=[1, 3], default=1, help="number of nearest neighbors to use")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

# create a dictionary of list objects equal to the number of neighbors to test
neighbors_dict = {}
for i in range(1, args.neighbors + 1):
    neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : 0}

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
    # plot training
    for i in range(0, len(dTrainingData)):
        if dTrainingData[i][2] == 1:
            plt.plot(dTrainingData[i][0], dTrainingData[i][1], dVariables['dColors'][1][0])
        else:
            plt.plot(dTrainingData[i][0], dTrainingData[i][1], dVariables['dColors'][2][0])
    # plot testing
    if variables_dict['majority_type'] == 1 or (vTestData[2] == 1 and variables_dict['majority_type'] == 0):
        plt.plot(vTestData[0], vTestData[1], dVariables['dColors'][1][1])
    else:
        plt.plot(vTestData[0], vTestData[1], dVariables['dColors'][2][1])
    # if plotting neighbors
    if dNeighbors[1]['index'] >= 0:
        # plot neighbors
        for i in range(1, len(dNeighbors) + 1):
            if dTrainingData[dNeighbors[i]['index']][2] == 1:
                plt.plot(dTrainingData[dNeighbors[i]['index']][0], dTrainingData[dNeighbors[i]['index']][1], dVariables['dColors'][3][0])
            else:
                plt.plot(dTrainingData[dNeighbors[i]['index']][0], dTrainingData[dNeighbors[i]['index']][1], dVariables['dColors'][3][1])

    plt.title(dVariables['plot_title'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# variables that are useful to pass around
variables_dict = {
    'training_file' : "./data/homework_classify_train_2D.dat"
    , 'testing_file' : "./data/homework_classify_test_2D.dat"
    , 'plot_title' : 'Default Title'
    , 'majority_type' : 0
    , 'classification' : 'UNK'
    , 'dColors' : { # plotting shapes and colors
        1 : ['k.', 'ko']    # black . data and blue o mean
        , 2 : ['g.', 'go']  # green + data and red "filled +" mean
        , 3 : ['ks', 'gs']  # cyan x data and magenta "filled x" mean
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
    distances = []  # for debugging only (store then sort all distances for comparison to the chosen distances)
    test_2d_vec = (vTestData[0], vTestData[1]) # does not include the "type" column
    # Loop through the training set to find the least distance(s)
    for i in range(0, len(dTrainingData)):
        train_2d_vec = (dTrainingData[i][0], dTrainingData[i][1]) # does not include the "type" column
        EuclideanDistance = EuclideanDistanceBetweenTwoVectors(test_2d_vec, train_2d_vec)
        distances.append(EuclideanDistance) # for debugging only
        neighbor_max_index = -1
        neighbor_max_value = -1
        # Loop through the neighbors dict so the maximum stored is always replaced first
        for j in range(1, len(dNeighbors) + 1):
            if dNeighbors[j]['distance'] > neighbor_max_value:
                neighbor_max_value = dNeighbors[j]['distance']
                neighbor_max_index = j
        # save the newest least distance over the greatest existing neighbor distance
        if EuclideanDistance < neighbor_max_value:
            dNeighbors[neighbor_max_index]['distance'] = EuclideanDistance
            dNeighbors[neighbor_max_index]['index'] = i
            dNeighbors[neighbor_max_index]['type'] = dTrainingData[i][2]

    # debugging: print the least distances out of all distances calculated
    if args.verbosity > 1:
        distances.sort()
        print("least distances:")
        for i in range(0, len(dNeighbors)):
            print(f"min{i}:({distances[i]}) \t& neighbors:({dNeighbors[i+1]['distance']})")

# Return the "type" value (1 or 2)
def GetNearestNeighborMajorityType(dNeighbors):
    type_1_count = 0
    type_2_count = 0
    for i in range(1, len(dNeighbors) + 1):
        if dNeighbors[i]['type'] == 1:
            type_1_count += 1
        elif dNeighbors[i]['type'] == 2:
            type_2_count += 1
    if type_1_count > type_2_count:
        return 1
    else:
        return 2

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
    print("The testing samples:")
    for i in range(0, len(testing_dict)):
        print(f"\t{i} {testing_dict[i]}")

# loop through all testing data
for i in range(0, len(testing_dict)):
    # create the test vector to pass around
    test_vec = (testing_dict[i][0], testing_dict[i][1], testing_dict[i][2])

    # create the plot graph
    variables_dict['plot_title'] = "{} - Test type {}".format(i+1, testing_dict[i][2])
    Scatterplot2D(training_dict, test_vec, neighbors_dict, variables_dict)

    # get the k-nearest neighbors
    PopulateNearestNeighborsDicOfIndexes(neighbors_dict, training_dict, test_vec)

    # get the k-nearest neighbors' majority type
    variables_dict['majority_type'] = GetNearestNeighborMajorityType(neighbors_dict)

    # plot the chosen neighbors and the majority type of the current test vector
    if testing_dict[i][2] == variables_dict['majority_type']:
        variables_dict['classification'] = 'Correct'
    else:
        variables_dict['classification'] = 'Incorrect'
    variables_dict['plot_title'] = "{} - Test type {} & Most neighbors {}: {}".format(i+1, testing_dict[i][2], variables_dict['majority_type'], variables_dict['classification'])
    Scatterplot2D(training_dict, test_vec, neighbors_dict, variables_dict)

    # reset variables
    variables_dict['majority_type'] = 0
    variables_dict['classification'] = 'UNK'
    for i in range(1, args.neighbors + 1):
        neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : 0}

