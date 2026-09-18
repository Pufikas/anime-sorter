import os
import shutil
import json
import onnxruntime as ort
ort.preload_dlls(directory="")

from imgutils.tagging import get_wd14_tags

dir_path = "test/input"
unknown_char_path = "test/unknown"
recognized_path = "test/output"
backup_path = "test/backup"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

with open("franchises.json", "r", encoding="utf-8") as f:
    franchise_data = json.load(f)

FRANCHISES = {
    character: franchise # "hakurei_reimu": "touhou"
    for franchise, characters in franchise_data.items() # get all characters in touhou
    for character in characters # loop in touhou characters
}

def list_files(path="."):
    for e in os.listdir(path):
        full_path = os.path.join(path, e)
        
        if os.path.isdir(full_path):
            list_files(full_path)
        elif os.path.splitext(full_path)[1].lower() in IMAGE_EXTENSIONS:
            get_character(full_path)

def get_character(file):
    rating, features, characters = get_wd14_tags(file)

    # it's possible that the model will throw `characters = {}`
    if not characters:
        move_to_location(file, unknown_char_path)
        return

    character, confidence = max(
        characters.items(), # creates pairs => ("hakurei_reimu", 0.91) ("kirisame_marisa", 0.23) etc..
        key = lambda item: item[1] # for each item use the second element (0,91)
    )

    if confidence >= 0.80:
        character_path = group_to_franchise(character)

        copy_to_location(file, character_path)
        move_to_location(file, backup_path)
    else:
        move_to_location(file, unknown_char_path)
    

    print(file, character, "{:.2f}".format(confidence))

def group_to_franchise(character):
    parts = character.split("_(") # artoria_pendragon(fate) => ['artoria_pendragon', 'fate)']

    if len(parts) == 2:
        character_name = format_name(parts[0])
        franchise = format_name(parts[1].rstrip(")")) # gets franchise from the file name tag

        return os.path.join(franchise, character_name)
    
    # from the local franchise list group character
    franchise = FRANCHISES.get(character)

    if franchise:
        return os.path.join(format_name(franchise), format_name(character))

    return format_name(character)

def create_folder(path):
    os.makedirs(path, exist_ok=True)

# returns, hiiragi_kagami => Hiiragi_Kagami, marnie => Marnie
def format_name(name):
    return "_".join(part.capitalize() for part in name.split("_"))

def move_to_location(file, location):
    create_folder(location)

    destination = os.path.join(
        location, 
        os.path.basename(file)
    )

    shutil.move(file, destination)

# copies file to recongnized character folder, else moves to unknown character folder
def copy_to_location(file, character=None):
    if character:
        path = os.path.join(recognized_path, character)
    else:
        path = unknown_char_path

    create_folder(path)

    destination = os.path.join(
        path, 
        os.path.basename(file)
    )

    shutil.copy2(file, destination)

list_files(dir_path)
