from fastapi import UploadFile
import cv2
import io
import numpy as np

def process_image(img: UploadFile):
    ioBuffer = img.file.read()
    image = cv2.imdecode(np.frombuffer(ioBuffer, np.uint8), 1)
    return image
