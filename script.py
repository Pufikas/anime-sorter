import os
from imgutils.tagging import get_wd14_tags

dir_path = "test/"

def list_files(path="."):
    for e in os.listdir(path):
        full_path = os.path.join(path, e)
        
        if os.path.isdir(full_path):
            list_files(full_path)
        else:
            get_character(full_path)

def get_character(path):
    rating, features, characters = get_wd14_tags(path)
    print(path, characters)


def create_folder(path):
    # create folder if needed at path

def copy_to_location(path):
    # copies the image to the path location

list_files(dir_path)
