import os
import glob
import pickle
from annoy import AnnoyIndex
from .SkywatchAI import get_faces, align_face, get_face_embedding

embedding_size = 128

def build_db(face_path, save_path):
    """
    Builds FaceEmbedding Database of people

    Args:
        face_path (Path): Face Directory Path
        save_path ([type]): Save Path for SkywatchDB
    """
    
    face_tree = AnnoyIndex(embedding_size, 'euclidean')
    image_paths = _get_image_paths(face_path)
    i = 1
    person_id_map = {}
    for person, images in image_paths.items():
        for image in images:
            faces = get_faces(image, enforce=True)
            try:
                aligned_face = align_face(faces[0]['image'])
                embedding = get_face_embedding(aligned_face)
            except IndexError:
                raise AssertionError('Could not detect face in '+ image)
            except TypeError:
                raise TypeError(f"Got an null object for {person} and {image}")
            face_tree.add_item(i, embedding)
            person_id_map[i] = person
            i += 1
    face_tree.build(5)
    try:
        face_tree.save(save_path+'faceEmbed.db')
        save_file = open(save_path+'nameMap.db', 'wb')
        pickle.dump(person_id_map, save_file)
        save_file.close()
        print('Skywatch Database successfully saved !')
    except:
        raise SystemError('Cannot save data files')

def load_db(path):
    """
    Loads SkywatchDB and returns the Objects

    Args:
        path(Path) : Path to directory containing database

    Returns:
        annoy.tree : Facial Embedding Database Tree 
        dict       : Person Name to ID Map for Database Tree
    """
    try:
        face_tree = AnnoyIndex(embedding_size, 'euclidean')
        face_tree.load(path+'faceEmbed.db')
        f = open(path+'nameMap.db', 'rb')
        personMap = pickle.load(f)
        print('Skywatch Database has been successfully loaded and ready')
    except FileNotFoundError:
        raise FileNotFoundError("File cannot be reached, check the file path")
    return face_tree, personMap


def _get_people_names(dir):
    try:
        names = os.listdir(dir)
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find the image in directories. \
                Please check with the instruction on how to model the folder")
    if '.gitkeep' in names:
        names.remove('.gitkeep')
        return names
    return names

def _get_image_paths(dir):
    names = _get_people_names(dir)
    image_data = {}
    for name in names:
        person_img_path = os.path.join(dir, name)
        image_data[name] = glob.glob(person_img_path+'/*.jpg')
        image_data[name].extend(glob.glob(person_img_path+'/*.png'))
    return image_data
