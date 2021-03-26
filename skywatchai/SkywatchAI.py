import cv2
import math
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from .model import FaceNet
from .utils import getEuclideanDistance, read_img, l2_normalization, preprocess_image


detector = MTCNN()
embedder = FaceNet.load_model()
threshold = 0.80
unknown_token = 'Unidentified'


input_shape = embedder.layers[0].input_shape
try:
    image_shape = input_shape[0][1:3]
except:
    image_shape = input_shape[1:3]

def compare(img1, img2):
    """
    Compare two images of people's face and returns and verify whether they are same.

    Args:
        img1 (str, cv2.Image, np.array): Path or Image Object
        img2 (str, cv2.Image, np.array): Path or Image Object

    Returns:
        boolean: Whether it is same person.
    """
    emb1 = get_face_embedding(img1)
    emb2 = get_face_embedding(img2)
    dist = getEuclideanDistance(emb1, emb2)
    return True if dist < threshold else False

def detect_faces(img, enforce=True):
    """
    Runs facial detection and marks landmarks and bounding box of a person's face. Returns the annotated image.

    Args:
        img (str, cv2.Image, np.array): Input Image
        enforce (bool, optional): Enforce Detection and throws in error if not found. Defaults to True.

    Returns:
        cv2.Image : Annotated Image
    """
    faces = get_faces(img, enforce=enforce)
    if not type(img) == np.ndarray:
        img = read_img(img)
    for face in faces:
        if face['confidence'] > 0.1:
            bbox = face['box']
            keypoints = face['keypoints']
            cv2.rectangle(img,
              (bbox[0], bbox[1]),
              (bbox[0]+bbox[2], bbox[1] + bbox[3]),
              (36,255,12),
              2)
            cv2.circle(img,(keypoints['left_eye']), 2, (36,255,12), 2)
            cv2.circle(img,(keypoints['right_eye']), 2, (36,255,12), 2)
            cv2.circle(img,(keypoints['nose']), 2, (36,255,12), 2)
            cv2.circle(img,(keypoints['mouth_left']), 2, (36,255,12), 2)
            cv2.circle(img,(keypoints['mouth_right']), 2, (36,255,12), 2)
    return img
    
def get_faces(img, enforce=True):
    """
    Returns a list of cropped facial images along with it's landmark information given from MTCNN model.

    Args:
        img (str, cv2.Image, np.array): Image to extract the cropped faces
        enforce (bool, optional): [description]. Defaults to True.

    Returns:
        list: List of Dictionary with cropped face with other informations
    """
    if not type(img) == np.ndarray:
        img = read_img(img)
    faces = []
    detections = detector.detect_faces(img)
    if len(detections) > 0:
        for face in detections:
            x, y, w, h = face['box']
            cropped_face = img[int(y): int(y+h), int(x) : int(x+w)]
            face['image'] = cropped_face
            faces.append(face)
        return faces
    else:
        if enforce != True:
            return img
        raise ValueError('Face could not be detected please check the image.')


def get_face_embedding(img):
    """
    Returns Embedding Representation of one's face.

    Args:
        img (str, cv2.Image, np.array): Cropped and aligned image of a person's face

    Returns:
        np.array : Embedding Vector (128-Dim) of a face
    """
    if not type(img) == np.ndarray:
        img = read_img(img)
    processed_img = preprocess_image(img, image_shape)
    embedding = l2_normalization(embedder.predict(processed_img)[0, :])
    return embedding

def align_face(img):
    """
    Aligns a person's face image for better face recognition.

    Args:
        img(cv2.Image, np.array) : Face Image of a Person

    Returns:
        cv2.Image : face aligned image of a given image
    """
    detection = detector.detect_faces(img)
    if len(detection)>0:
        detection = detection[0]
        keypoints = detection['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
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
    

def get_people(image, embeds_db, name_map, enforce=True):
    """
    It returns labelled cropped face image of people in an given image.

    Args:
        image (str or numpy array): Input Image
        embeds_db (annoy tree): FaceDB Object
        name_map (dict): Key to name map
        enforce (bool, optional): Whether to enforce detection. Defaults to True.

    Returns:
        list: list of facial details of each person
    """
    faces = get_faces(image, enforce=enforce)
    found_people = []
    for face in faces:
        if face['image'].size != 0:
            emb = get_face_embedding(face['image'])
            result = embeds_db.get_nns_by_vector(emb, 1, include_distances=True)
            id, distance = result
            if distance[0] < 0.8:
                face['name'] = name_map[id[0]]
            else:
                face['name'] = unknown_token
            found_people.append(face)
    return found_people

def find_people(img, embeds_db, name_map, enforce=True):
    """
    Finds people in image, annotate them with their names and bounding box if they are known.

    Args:
        image (str or numpy array): Input Image
        embeds_db (annoy tree): FaceDB Object
        name_map (dict): Key to name map
        enforce (bool, optional): Whether to enforce detection. Defaults to True.

    Returns:
        cv2.Image : Annotated Image
    """
    faces = get_people(img, embeds_db, name_map, enforce)
    img = read_img(img)
    for face in faces:
        if face['confidence'] > 0.1:
            bbox = face['box']
            cv2.rectangle(img,
                (bbox[0], bbox[1]),
                (bbox[0]+bbox[2], bbox[1] + bbox[3]),
                (36,255,12),
                2)
            cv2.putText(img, face['name'], (bbox[0], bbox[1]-1), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36,255,12))
    return img