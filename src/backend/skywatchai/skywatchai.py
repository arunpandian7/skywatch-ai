import .MTCNN
import .FaceNet
from .utils import getEuclideanDistance

import numpy as np
import math
from PIL import Image

def align_face(img, left_eye, right_eye):
    # Alignment is done by rotation with respect to Left and Right Eye
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # Finding the direction of rotation
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate clockwise   
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate counter-clockwise

    # Find lenght of Right Angled Triangle
    a = getEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = getEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = getEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------
	# Applying Cosine Law
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
        
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
        if direction == -1:
            angle = 90 - angle
        # Rotating Given Image
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    return img

def crop_face(img):
    detector = MTCNN() 
    detections = detector.detect_faces(img) # Detects Face 

    if len(detections) > 0:
        detection = detections[0]
        keypoints = detection["keypoints"]
        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]
        
        # Align image w.r.t to Face
        img = align_face(img, left_eye, right_eye)
        # Detect Face Image on Aligned Image to crop the Face Portion
        detections = detector.detect_faces(img)
        x, y, w, h = detection['box']
        cropped_face = img[int(y): int(y+h), int(x) : int(x+w)]
        return cropped_face
    else:
        raise ValueError('Face could not be detected please check the image.')

