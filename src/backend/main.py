"""
    Dirver Python File for Skywatch AI - an Face Recogniton based Attendance Monitor
"""

__author__ = "Arun Pandian R"
__version__ = "0.0.1"
__license__ = "MIT"

import argparse
import os
import glob

from numpy.lib.type_check import imag
from skywatchai.SkywatchAI import get_face_embedding

def get_names(dir):
    try:
        names = os.listdir(dir)
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find the image in directories. \
                Please check with the instruction on how to model the folder")
    if '.gitkeep' in names:
        names.remove('.gitkeep')
        return names
    return names

def get_person_images(dir, names):
    image_data = {}
    for name in names:
        person_img_path = os.path.join(dir, name)
        image_data[name] = glob.glob(person_img_path+'/*.jpg')
        image_data[name].extend(glob.glob(person_img_path+'/*.png'))
    return image_data

def create_embedding_data(dir, names):
    image_data = get_person_images(dir, names)
    embedding_data = {}
    for person in image_data.keys():
        embedding_data[person] = list()
        for image in image_data[person]:
            embedding = get_face_embedding(image)
            embedding_data[person].append(embedding)
    return embedding_data

def main(args):
    """ Main entry point of the app """
    if args.mode == 'transform' and args.directory != None:
        name_list = get_names(args.directory)
        print(create_embedding_data(args.directory, name_list))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required argument to execute the function
    parser.add_argument("mode", help="Pass the keyword to execute")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--dir", action="store", dest="directory")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="SkywatchAI - version {version}".format(version=__version__))

    args = parser.parse_args()
    main(args)


