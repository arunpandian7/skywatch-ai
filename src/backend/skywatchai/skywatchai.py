from io import FileIO
import os
import glob
import pickle
from annoy import AnnoyIndex
from mtcnn.mtcnn import MTCNN
from .preprocessing import align_face, get_face_embedding, get_faces
from .model.FaceNet import load_facenet
from .utils import getEuclideanDistance

def get_people_names(dir):
    try:
        names = os.listdir(dir)
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find the image in directories. \
                Please check with the instruction on how to model the folder")
    if '.gitkeep' in names:
        names.remove('.gitkeep')
        return names
    return names

def get_image_paths(dir):
    names = get_people_names(dir)
    image_data = {}
    for name in names:
        person_img_path = os.path.join(dir, name)
        image_data[name] = glob.glob(person_img_path+'/*.jpg')
        image_data[name].extend(glob.glob(person_img_path+'/*.png'))
    return image_data

class FaceID():
    def __init__(self, dir, enforce_detection=True) -> None:
        self.detector = MTCNN()
        self.emb_model = load_facenet()
        self.threshold = 0.80
        self.enforce_detection = enforce_detection
        self.dir = dir
        self.map_save_path = os.path.join('../../database', 'personMap.db')
        self.emb_save_path = os.path.join('../../database', 'face_embeds.ann')
        input_shape = self.emb_model.layers[0].input_shape
        self.embedding_size = 128
        self.unknownName = 'Unknown'
        try:
            self.image_size = input_shape[0][1:3]
        except:
            self.image_size = input_shape[1:3]
    
    def verify(self, img1, img2):
        emb1 = get_face_embedding(img1, self.emb_model, self.image_size)
        emb2 = get_face_embedding(img2, self.emb_model, self.image_size)
        dist = getEuclideanDistance(emb1, emb2)
        return True if dist < self.threshold else False
    
    def build_face_db(self):
        face_tree = AnnoyIndex(self.embedding_size, 'euclidean')
        image_paths = get_image_paths(self.dir)
        i = 1
        person_id_map = {}
        for person, images in image_paths.items():
            for image in images:
                faces = get_faces(image, self.detector, enforce=True)
                aligned_face = align_face(faces[-1]['image'], self.detector)
                embedding = get_face_embedding(aligned_face, self.emb_model, self.image_size)
                face_tree.add_item(i, embedding)
                person_id_map[i] = person
                i += 1
        face_tree.build(5)
        print('ANN Tree Built Successfully')
        try:
            face_tree.save(self.emb_save_path)
            save_file = open(self.map_save_path, 'wb')
            pickle.dump(person_id_map, save_file)
            save_file.close()
            print('Files saved successfully in the drive')
        except:
            raise SystemError('Cannot save data files')
    
    def load_face_db(self):
        try:
            self.face_tree = AnnoyIndex(self.embedding_size, 'euclidean')
            self.face_tree.load(self.emb_save_path)
            f = open(self.map_save_path, 'rb')
            self.personMap = pickle.load(f)
            print('Database has been successfully loaded and ready')
        except FileNotFoundError:
            raise FileNotFoundError("File cannot be reached, check the file path")

    def find_people(self, image):
        faces = get_faces(image)
        found = []
        for face in faces:
            emb = get_face_embedding(face['image'], self.emb_model, self.image_size)
            result = self.face_tree.get_nns_by_vector(emb, 1, include_distances=True)
            id, distance = result
            if distance < 0.8:
                face['name'] = self.personMap[id]
            else:
                face['name'] = self.unknownName
            found.append(face)
        return found
