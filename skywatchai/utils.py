import numpy as np
import cv2

def preprocess_image(img, target_size):
    processed_img = cv2.resize(img, target_size)
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = processed_img / 255

    return processed_img

def getEuclideanDistance(a, b):
    euclidean_distance = a - b
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalization(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def read_img(path):
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except:
        raise AssertionError('Could not read image in +'+path)