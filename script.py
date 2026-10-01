import os
from collections import Counter, defaultdict
import shutil
import json
import onnxruntime as ort
from tqdm import tqdm
ort.preload_dlls(directory="")

from imgutils.tagging import get_wd14_tags
from datasets import load_dataset

dataset = load_dataset(
    "tirta123/noob-wiki",
    split="train"
)

with open("settings.json", "r", encoding="utf-8") as f:
    franchise_data = json.load(f)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SETTINGS = franchise_data.get("settings", {})
CONFIDENCE = SETTINGS.get("CONFIDENCE", 0.80)
CREATE_CHARACTER_FOLDER = SETTINGS.get("CREATE_CHARACTER_FOLDER", True)
REMOVE_EMPTY_FOLDERS = SETTINGS.get("REMOVE_EMPTY_FOLDERS", True)
CHARACTER_FOLDER_MIN_COUNT = SETTINGS.get("CHARACTER_FOLDER_MIN_COUNT", 20)

INPUT_PATH = SETTINGS.get("INPUT_PATH")
UNKNOWN_PATH = SETTINGS.get("UNKNOWN_PATH")
OUTPUT_PATH = SETTINGS.get("OUTPUT_PATH")
BACKUP_PATH = SETTINGS.get("BACKUP_PATH")

ALIASES = franchise_data.get("overrides", {})
CUSTOM_GROUPS = franchise_data.get("franchise_groups", {})
INVALID_SYMBOLS = ['*', '"', '/', '\\', '<', '>', ':', '|', '?']

DATASET_FRANCHISES = {
    row["character"]: row["copyright"]
    for row in dataset
}

results = [] # sorting results

def list_files(path="."):
    files = []
    for e in os.listdir(path):
        full_path = os.path.join(path, e)
        
        if os.path.isdir(full_path):
            files.extend(list_files(full_path))
            
        elif os.path.splitext(full_path)[1].lower() in IMAGE_EXTENSIONS:
            files.append(full_path)

    return files

def clean_file_name(file):
    for sym in INVALID_SYMBOLS:
        file = file.replace(sym, "_")

    return file

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            full_path = os.path.join(root, directory)

            if not os.listdir(full_path):
                os.rmdir(full_path)

def get_character(file):
    rating, features, characters = get_wd14_tags(file)

    # it's possible that the model will throw `characters = {}`
    if not characters:
        results.append({
            "file": file,
            "character": None,
            "confidence": 0,
            "franchise": None,
        })

        return

    character, confidence = max(
        characters.items(), # creates pairs => ("hakurei_reimu", 0.91) ("kirisame_marisa", 0.23) etc..
        key = lambda item: item[1] # for each item use the second element (0,91)
    )

    group_confidence(character, confidence, file)

def group_confidence(character, confidence, file):
    if confidence < CONFIDENCE:
        results.append({
            "file": file,
            "character": character,
            "confidence": confidence,
            "franchise": None,
        })
        return

    franchise, character_name = get_franchise(character)

    results.append({
        "file": file,
        "character": character_name,
        "confidence": confidence,
        "franchise": franchise,
    })
            
def get_franchise(character):
    character = ALIASES.get(character, character)
    # get only character name with no franchise WD14 tag (mona_(genshin_impact) => mona)
    character_name = character.split("_(")[0]

    # user defined franchise
    for franchise, characters in CUSTOM_GROUPS.items():
        if character_name in characters:
            return franchise, character_name

    # tirta123 dataset franchise
    franchise = DATASET_FRANCHISES.get(character)

    if franchise:
        return franchise, character_name

    # WD14 default group
    parts = character.split("_(")

    if len(parts) == 2:
        franchise = parts[1].rstrip(")")
        return franchise, character_name

    # no franchise found
    return None, character_name

def group_to_franchise(franchise, character, character_count):
    c = clean_file_name(format_name(character))

    if franchise is None:
        return c

    f = clean_file_name(format_name(franchise))
    
    if (CREATE_CHARACTER_FOLDER and character_count >= CHARACTER_FOLDER_MIN_COUNT):
        return os.path.join(f, c)

    return f

def create_folder(path):
    os.makedirs(path, exist_ok=True)

# returns, hiiragi_kagami => Hiiragi_Kagami, marnie => Marnie
def format_name(name):
    return "_".join(part.capitalize() for part in name.split("_"))

def move_to_location(file, location):
    create_folder(location)
    cleaned_name = clean_file_name(os.path.basename(file))

    destination = os.path.join(
        location, 
        cleaned_name
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
    cleaned_name = clean_file_name(os.path.basename(file))

    destination = os.path.join(
        path, 
        cleaned_name
    )

    shutil.copy2(file, destination)

    return destination

def finalize():
    character_counts = Counter(
        result["character"]
        for result in results
        if result["character"] is not None
    )

    for result in results:
        file = result["file"]
        character = result["character"]
        confidence = result["confidence"]
        franchise = result["franchise"]

        # no char or low confidence moves to unknown path
        if character is None or confidence < CONFIDENCE:
            destination = move_to_location(file, UNKNOWN_PATH)
            continue

        character_path = group_to_franchise(
            result["franchise"],
            character,
            character_counts[character]
        )

        destination = copy_to_location(file, character_path)
        move_to_location(file, BACKUP_PATH)

    if REMOVE_EMPTY_FOLDERS:
        remove_empty_folders(INPUT_PATH)

    print_results()

def print_results():
    recognized = 0
    unknown = 0
    franchises = defaultdict(Counter)

    for result in results:
        character = result["character"]
        confidence = result["confidence"]

        if character is None or confidence < CONFIDENCE:
            unknown += 1
            continue

        recognized += 1

        franchise = result["franchise"] or None
        franchises[franchise][character] += 1

    print(f"\nRecognized         {recognized}")
    print(f"Unknown            {unknown}\n")

    for franchise, characters in franchises.items():
        print("\n", franchise)

        for character, count in characters.most_common():
            print(f"  {character:<25} {count}")

files = list_files(INPUT_PATH)

for file in tqdm(files, desc="Analyzing", unit="image"):
    get_character(file)
    
finalize()
