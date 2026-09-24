import os
import shutil
import json
import onnxruntime as ort
ort.preload_dlls(directory="")

from imgutils.tagging import get_wd14_tags
from datasets import load_dataset

dataset = load_dataset(
    "tirta123/noob-wiki",
    split="train"
)

# configure these as you see fit
INPUT_PATH = "sorter/input" # input folder to process images from
UNKNOWN_PATH = "sorter/unknown" # not recognized images
OUTPUT_PATH = "sorter/output" # recognized and tagged images folder output
BACKUP_PATH = "sorter/backup" # path to backup images from INPUT_PATH
REMOVE_EMPTY_FOLDERS = True # should it remove empty folders from INPUT_PATH?

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

with open("franchises.json", "r", encoding="utf-8") as f:
    franchise_data = json.load(f)

ALIASES = franchise_data.get("overrides", {})
CUSTOM_GROUPS = franchise_data.get("franchise_groups", {})

DATASET_FRANCHISES = {
    row["character"]: row["copyright"]
    for row in dataset
}

def list_files(path="."):
    for e in os.listdir(path):
        full_path = os.path.join(path, e)
        
        if os.path.isdir(full_path):
            list_files(full_path)
            
            if not os.listdir(full_path) and REMOVE_EMPTY_FOLDERS: # remove empty dir
                os.rmdir(full_path)

        elif os.path.splitext(full_path)[1].lower() in IMAGE_EXTENSIONS:
            get_character(full_path)

def get_character(file):
    rating, features, characters = get_wd14_tags(file)

    # it's possible that the model will throw `characters = {}`
    if not characters:
        destination = move_to_location(file, UNKNOWN_PATH)
        print(f"[UNKNOWN] {destination} -> No character detected")
        return

    character, confidence = max(
        characters.items(), # creates pairs => ("hakurei_reimu", 0.91) ("kirisame_marisa", 0.23) etc..
        key = lambda item: item[1] # for each item use the second element (0,91)
    )

    if confidence >= 0.80:
        character_path = group_to_franchise(character)

        destination = copy_to_location(file, character_path)
        move_to_location(file, BACKUP_PATH)

        print(f"[OK] {destination} -> {character} ({confidence:.2f})")
    else:
        destination = move_to_location(file, UNKNOWN_PATH)
        print(f"[UNKNOWN] {destination} -> {character} ({confidence:.2f})")
    
def group_to_franchise(character):
    character = ALIASES.get(character, character)

    # user defined franchise
    for franchise, characters in CUSTOM_GROUPS.items():
        if character in characters:
            return os.path.join(
                format_name(franchise),
                format_name(character)
            )
    
    # tirta123 dataset franchise
    franchise = DATASET_FRANCHISES.get(character)

    if franchise:
        character_name = character.split("_(")[0]

        return os.path.join(
            format_name(franchise),
            format_name(character_name)
        )

    # WD14 default group
    parts = character.split("_(")

    if len(parts) == 2:
        character_name = parts[0]
        franchise = parts[1].rstrip(")")

        return os.path.join(
            format_name(franchise),
            format_name(character_name)
        )
    
    # no franchise found
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
    
    return destination

# copies file to recongnized character folder, else moves to unknown character folder
def copy_to_location(file, character=None):
    if character:
        path = os.path.join(OUTPUT_PATH, character)
    else:
        path = UNKNOWN_PATH

    create_folder(path)

    destination = os.path.join(
        path, 
        os.path.basename(file)
    )

    shutil.copy2(file, destination)

    return destination

list_files(INPUT_PATH)
