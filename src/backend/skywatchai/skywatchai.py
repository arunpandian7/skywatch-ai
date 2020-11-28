import os
import glob
import pickle
from annoy import AnnoyIndex
from .model.FaceNet import loadmodel
from .preprocessing import get_face_embedding
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
        self.emb_model = loadmodel()
        self.threshold = 0.80
        self.enforce_detection = enforce_detection
        self.dir = dir
        self.emb_save_path = os.path.join(dir, 'face_embeds.ann')
        input_shape = self.emb_model.layers[0].input_shape
        self.embedding_size = 128
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
        for person, images in image_paths:
            for image in images:
                embedding = get_face_embedding(image, self.emb_model, self.image_size)
                face_tree.add_item(i, embedding)
                person_id_map[i] = person
                i += 1
        face_tree.build(5)
        try:
            face_tree.save(self.emb_save_path)
            save_file = open('../../database/personMap.pickle', 'wb')
            pickle.dump(person_id_map, save_file)
            save_file.close()
        except:
            SystemError('Cannot save data files')
    
    


            



