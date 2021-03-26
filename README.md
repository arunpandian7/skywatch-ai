# Skywatch.ai - Facial Recognition System
Skywatch.ai is an API wrapper for powerful face detection and recognition models. It enables an efficient and clean way to use these models in your project without having to worry about the backend clusters. The focus of these project is to provide a easy-to-use package that speeds up development of lot of applications and research.

## Usage
You can install skywatchai using [PyPi](https://pypi.org/project/skywatchai/)
```
pip install skywatchai
```

### Face Detection
```python
import skywatchai.SkywatchAI as skai
img_path = 'test/oscar.jpg'
detected = skai.detect_faces(img_path)
```
<img src="https://github.com/arunpandian7/skywatch-ai/blob/master/asset/face-detect.png" width="60%" align="center" />

### Face Verification
```python
img1 = 'test/image1.jpg'
img2 = 'test/image2.jpg' 
result = skai.compare(img1, img2)
print('Are they same person:', result)
```
![Face Verification Result](https://github.com/arunpandian7/skywatch-ai/blob/master/asset/face-verify-1.png)

### Face Recognition
#### Building the Database
For Face Recognition, you need to build the repository of people face images. You can have multiple images of same person under the directory of his name. Please refer to the below directory tree.
```
database
├── people
|   ├── Brad Pitt
|   |   ├── image1.jpg
|   |   ├── image2.jpg
|   |   ├── ..
|   ├── Bradley Cooper
|   |   ├── image1.jpg
|   |   ├── image2.jpg
|   |   ├── ..
|   ├── Chris Hemsworth
|   |   ├── image1.jpg
|   |   ├── image2.jpg
|   |   ├── ..
|   ├── ..
```

```python
import skywatchai.SkywatchDB as skdb
# Give face_path directing to folder containing images following the above requirement
skdb.build_db(face_path='database/people/', save_path='database/')
faceDB, nameMap = skdb.load_db(path='database/')
```
#### Recognizing the person from Database
```python
annot_img = skai.find_people(img, faceDB, nameMap)
```
<img src="https://github.com/arunpandian7/skywatch-ai/blob/master/asset/face-recogn.png" width="60%" align="center" />

## Dependencies
- [MTCNN](https://pypi.org/project/mtcnn/)
- [FaceNet](https://github.com/davidsandberg/facenet)
- Tensorflow
- Keras
- Numpy
- OpenCV

## Acknowledgement
I am very thankful for [deepface](https://github.com/serengil/deepface) library created by [Sefin Seringil](https://sefiks.com). His works were very useful for me in creating this project.