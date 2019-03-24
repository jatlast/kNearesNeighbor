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
parser = argparse.ArgumentParser(description="perform the k-means clustering on 1 to 2-dimensional data")
parser.add_argument("-f", "--filename", default="./data/homework_classify_train_2D.dat", help="file name (and path if not in . dir)")
parser.add_argument("-n", "--neighbors", type=int, choices=[1, 3], default=1, help="number of nearest neighbors to use")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

# create a dictionary of list objects equal to the number of neighbors to test
neighbor_dict = {}
for i in range(1, args.neighbors + 1):
    neighbor_dict[i] = []

if args.verbosity > 0:
    print(f"filename={args.filename} : neighbors={args.neighbors} : len(neighbor_dict)={len(neighbor_dict)}")

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
def Scatterplot2D(dNeighbors, dMeans, dVariables):
    for i in range(1, len(dNeighbors) + 1):
        x = []
        y = []
        for j in range(0, len(dNeighbors[i])):
            x.append(dNeighbors[i][j][0])
            y.append(dNeighbors[i][j][1])
        plt.plot(x, y, dVariables['dColors'][i][0])
        plt.plot(dMeans[i][0], dMeans[i][1], dVariables['dColors'][i][1])
    plt.title(dVariables['plot_title'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

variables_dict = {
    'dColors' : { # plotting shapes and colors
        1 : ['k.', 'bo']    # black . data and blue o mean
        , 2 : ['g+', 'rP']  # green + data and red "filled +" mean
        , 3 : ['cx', 'mX']  # cyan x data and magenta "filled x" mean
    }
}

# read in the training data file
with open(args.filename, mode='r') as data_file:
    training_dict = {}
    line_number = 0
    # parse data file
    for line in data_file:
        line_number += 1
#        training_dict[line_number] = []
        # a regular expression to match both the 1-dimensional & 2-dimensional files supplied
        match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d)', line)
        # get the x (and possibly the y) values
        if match:
            if match.group(1) and match.group(2) and match.group(3):
                training_dict[line_number] = [float(match.group(1)), float(match.group(2)), int(match.group(3))]
            else:
                print(f"Warning: all three groups were not found on line ({line})")
        else:
            print(f"Warning: no match for line ({line})")

if args.verbosity > 1:
    print("The first 5 training samples:")
    for i in range(1, len(training_dict) + 1):
        if i > 5:
            break
        else:
            print(f"\t{i} {training_dict[i]}")
