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
DEBUG = SETTINGS.get("DEBUG", False)
CONFIDENCE = SETTINGS.get("CONFIDENCE", 0.80)
CREATE_CHARACTER_FOLDER = SETTINGS.get("CREATE_CHARACTER_FOLDER", True)
REMOVE_EMPTY_FOLDERS = SETTINGS.get("REMOVE_EMPTY_FOLDERS", True)
CHARACTER_FOLDER_MIN_COUNT = SETTINGS.get("CHARACTER_FOLDER_MIN_COUNT", 20)
MULTI_CHARACTER_CONFIDENCE = SETTINGS.get("MULTI_CHARACTER_CONFIDENCE", 0.80)
MULTI_CHARACTER_MIN_COUNT = SETTINGS.get("MULTI_CHARACTER_MIN_COUNT", 2)

INPUT_PATH = SETTINGS.get("INPUT_PATH")
UNKNOWN_PATH = SETTINGS.get("UNKNOWN_PATH")
OUTPUT_PATH = SETTINGS.get("OUTPUT_PATH")
BACKUP_PATH = SETTINGS.get("BACKUP_PATH")
CREATE_BACKUPS = SETTINGS.get("CREATE_BACKUPS", False)

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

def normalize_characters(characters):
    normalized = {}

    for character, confidence in characters.items():
        character = ALIASES.get(character, character)

        if character not in normalized:
            normalized[character] = confidence
        else:
            normalized[character] = max(
                normalized[character],
                confidence
            )

    return normalized

def get_character(file):
    rating, features, characters = get_wd14_tags(file)

    # no data about the image
    if not characters:
        results.append({
            "file": file,
            "character": None,
            "confidence": 0,
            "franchise": None,
            "franchise_score": 0,
            "multi_character": False,
        })
        return

    # apply user manual overrides
    characters = normalize_characters(characters)
    # sums all characters from X franchise
    franchise_scores = get_franchise_scores(characters)
    
    if DEBUG:
        tqdm.write(str(characters))

    # single character highest confidence
    character, confidence = max(
        characters.items(),
        key = lambda item: item[1]
    )

    # only take high confidence characters
    high_confidence_characters = [
        character
        for character, confidence in characters.items()
        if confidence >= CONFIDENCE
    ]

    if len(high_confidence_characters) >= MULTI_CHARACTER_MIN_COUNT:
        if franchise_scores:
            # get highest franchise score
            franchise, franchise_score = max(
                franchise_scores.items(),
                key = lambda item: item[1]
            )
        else:
            franchise = None
            franchise_score = 0

        # character is none to avoid creating character folder
        results.append({
            "file": file,
            "character": None,
            "confidence": confidence,
            "franchise": franchise,
            "franchise_score": franchise_score,
            "multi_character": True,
        })
        return

    # single character
    franchise, character_name = get_franchise(character)

    results.append({
        "file": file,
        "character": character_name,
        "confidence": confidence,
        "franchise": franchise,
        "franchise_score": franchise_scores.get(franchise, 0),
        "multi_character": False,
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

def get_franchise_scores(characters):
    # if there are multiple characters returns the highest probability of what X franchise
    scores = defaultdict(float)

    for character, confidence in characters.items():
        franchise, _ = get_franchise(character)

        if franchise:
            scores[franchise] += confidence

    return scores

def group_to_franchise(franchise, character, character_count):
    clean = clean_file_name(format_name(character))

    if franchise is None:
        return clean

    franchise = clean_file_name(format_name(franchise))
    
    if (CREATE_CHARACTER_FOLDER and character_count >= CHARACTER_FOLDER_MIN_COUNT):
        return os.path.join(franchise, clean)

    return franchise

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
    # count only the high confidence characters
    character_counts = Counter(
        result["character"]
        for result in results
        if result["character"] is not None
        and result["confidence"] >= CONFIDENCE
    )

    for result in results:
        file = result["file"]
        character = result["character"]
        confidence = result["confidence"]
        franchise = result["franchise"]
        multiple = result["multi_character"]

        # multiple characters, move to franchise if any
        if multiple:
            if franchise:
                franchise_path = clean_file_name(format_name(franchise))

                move_to_location(file, os.path.join(OUTPUT_PATH, franchise_path))
            else:
                move_to_location(file, UNKNOWN_PATH)

            continue

        # no char or no franchise, low confidence moves to unknown path
        if character is None or franchise is None or confidence < CONFIDENCE:
            move_to_location(file, UNKNOWN_PATH)
            continue

        character_path = group_to_franchise(
            franchise,
            character,
            character_counts[character]
        )

        if CREATE_BACKUPS:
            copy_to_location(file, BACKUP_PATH)
        
        move_to_location(file, os.path.join(OUTPUT_PATH, character_path))
        

    if REMOVE_EMPTY_FOLDERS:
        remove_empty_folders(INPUT_PATH)

    print_results()

def print_results():
    recognized = 0
    unknown = 0
    multiple = Counter()
    franchises = defaultdict(Counter)

    for result in results:
        character = result["character"]
        confidence = result["confidence"]
        franchise = result["franchise"] or None

        if result["multi_character"]:
            multiple[franchise] += 1
            franchises[franchise] # init franchise so if a X franchise has only multi char images this would show from what franchise
            continue

        if character is None or confidence < CONFIDENCE:
            unknown += 1
            continue

        recognized += 1
        franchises[franchise][character] += 1

        

    print(f"\nRecognized         {recognized}")
    print(f"Unknown            {unknown}")
    print(f"Multiple           {sum(multiple.values())}\n")

    for franchise, characters in franchises.items():
        print(f"\n{franchise}")

        for character, count in characters.most_common():
            print(f"  {character:<25} {count}")

        if multiple[franchise]:
            print(f"  {'multiple':<25} {multiple[franchise]}")

files = list_files(INPUT_PATH)

for file in tqdm(files, desc="Analyzing", unit="image"):
    get_character(file)
    
finalize()
