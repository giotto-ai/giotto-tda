import math as m
import numpy as np

def rotate_clockwise(X):
    rotationMatrix = m.sqrt(2) / 2. * np.array([[1, -1], [1 , 1]])
    return np.dot(X, rotationMatrix)

def rotate_anticlockwise(X):
    rotationMatrix = m.sqrt(2) / 2. * np.array([[1, 1], [-1 , 1]])
    return np.dot(X, rotationMatrix)
