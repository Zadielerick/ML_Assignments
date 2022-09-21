#!/usr/bin/python3

import math
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import numpy
import matplotlib.pyplot as plt

def main():
    mean = [0,0]
    covariance_matrix = [[1,0],[0,1]]
    number_of_samples = 3000

    random_samples = n_dist_sample(mean, covariance_matrix, number_of_samples)
    x = []
    y = []
    for point in random_samples:
        x.append(point[0])
        y.append(point[1])

    print(random_samples)
    plt.scatter(x,y)
    plt.show()

    print(calc_eucledian_distance([1,0,0,1],[0,1,0,2]))
    print(calc_mahalanobis_distance(covariance_matrix, mean, [2,0]))

def n_dist_sample(mean, cov, num_samples):
    return multivariate_normal.rvs(mean, cov, num_samples)

def calc_discriminant_function():
    return 0

def calc_eucledian_distance(point1, point2):
    return distance.euclidean(point1, point2)

def calc_mahalanobis_distance(cov1, mean1, point1):
    point = numpy.array(point1)
    mean = numpy.array(mean1)
    cov = numpy.array(cov1)

    mean_difference = point - mean
    inv_cov = numpy.linalg.inv(cov)
    temp = numpy.dot(mean_difference, inv_cov)
    temp = numpy.dot(temp,mean_difference.T)
    return temp

if __name__ == "__main__":
    main()
