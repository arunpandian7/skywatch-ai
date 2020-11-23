import numpy as np
import math

def getEuclideanDistance(a, b):
    euclidean_distance = a - b
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance