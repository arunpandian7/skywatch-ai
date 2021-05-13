from ast import parse
import os
import glob
from os import path
import pickle
import pathlib

from skywatchai.utils import parse_path
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
    face_path = parse_path(face_path)
    save_path = parse_path(save_path)

    print("SkywatchDB Build Started...")
    
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
                print(f"Cannot detect face for {person} in {image}")
                continue
            face_tree.add_item(i, embedding)
            person_id_map[i] = person
            i += 1
    face_tree.build(5)
    try:
        face_tree.save(save_path.joinpath('faceEmbed.db').as_posix())
        save_file = open(save_path.joinpath('nameMap.db').as_posix(), 'wb')
        pickle.dump(person_id_map, save_file)
        save_file.close()
        print('SkywatchDB successfully saved at ', save_path.as_posix())
    except:
        raise SystemError('Storage Access Error. Cannot save Skywatch Database.')

def load_db(path):
    """
    Loads SkywatchDB and returns the Objects

    Args:
        path(Path) : Path to directory containing database

    Returns:
        annoy.tree : Facial Embedding Database Tree 
        dict       : Person Name to ID Map for Database Tree
    """
    path = parse_path(path)
    try:
        face_tree = AnnoyIndex(embedding_size, 'euclidean')
        face_tree.load(path.joinpath('faceEmbed.db').as_posix())
        f = open(path.joinpath('nameMap.db').as_posix(), 'rb')
        personMap = pickle.load(f)
        print('Skywatch Database has been successfully loaded and ready')
    except FileNotFoundError:
        raise FileNotFoundError("File cannot be reached, check the file path")
    return face_tree, personMap


def _get_people_names(dir:pathlib.Path):
    try:
        names = os.listdir(dir)
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find the image in directories")
    return names

def _get_image_paths(dir:pathlib.Path):
    names = _get_people_names(dir)
    image_data = {}
    for name in names:
        person_img_path = dir.joinpath(name)
        image_data[name] = [i for i in person_img_path.glob("*.jpg")]
        image_data[name].extend([i for i in person_img_path.glob("*.png")])
    return image_data
