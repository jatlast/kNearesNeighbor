# kNearestNeighbor
k-nearest neighbor classifier supporting k=1 &amp; k=3

```
Master's Degree: University of Michigan - Computer Science & Information Systems
Course: CSC 546 - Advanced Artificial Intelligence

Assignment Homework 7: Due: 03/26/2019 5:59 PM
    2.  k-nearest neighbor (5 points) – Write a k-NN classifier for the data 
        in the files provided. Make your training data from the training data
        files provided and test on the test file.

        a.  Test it with k=1
        b.  Test it with k=3
 
        Note:   Use 2-dimensional data in homework_classify_train_2D.dat and
                homework_classify_test_2D.dat.
```
## False Starts

1) 
2) 

## Final Solutions

1) 
2)

## Chosen Technologies

Motivation: Become more familiar with the following.
1) Artificial Intelligence supervised training algorithms for classification and prediction
2) Python's numpy & matplotlib.pyplot libraries for plot visualizations
3) Developing with Python 3.7 in Windows environment
4) IDE - Visual Studio Code
5) GitHub (I am becoming quite familiar)

## The following websites were referenced

* [Pyplot Tutorial](https://matplotlib.org/users/pyplot_tutorial.html) - For understanding the basics of matplotlib.pyplot
* [Specifying Colors](https://matplotlib.org/users/colors.html) - For pyplot color reference
* [matplotlib.markers](https://matplotlib.org/api/markers_api.html) - For pyplot marker type reference

### Prerequisites

- Python 3.6+

### Installing
```
Clone the "kNearestNeighbor" project into the desired directory.
kNearestNeighbor should run in any Python 3.6+ environment
```

### Command Line Specifications
```
> python kNearestNeighbor.py -h
--------------------------------------------------------------------
usage: kNearestNeighbor.py [-h] [-n {1,3}] [-v {0,1,2}]

Perform k-nearest neighbor classification on 2-dimensional data provided.
(Note: 2D file includes 3 columns: x, y, classification)

optional arguments:
  -h, --help            show this help message and exit
  -n {1,3}, --neighbors {1,3}
                        number of clusters to try
  -v {0,1,2}, --verbosity {0,1,2}
                        increase output verbosity
--------------------------------------------------------------------

Example> python kNearestNeighbor.py -v 2 -n 3
Results:    -v) verbosity is set to it most verbose setting of 2
            -n) the algorithm will attempt 3-nearest neighbor classification
```

### Plot Lable

Scatter Plot (0.90 > 0.20)
```
0.90 = maximum mean difference last calculated between the specified number of clusters
0.20 = mean difference that must be obtained before exiting the while loop
```

## License

This project is not licensed but feel free to play with any part you so desire.

## Acknowledgments

* matplotlib.org
* Google's vast doorway to every tid-bit of documentation on the internet
* All those wonderfully generous documentation writers and question answerers
