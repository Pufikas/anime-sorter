# anime-sorter

Detects anime characters from images and sorts them into folders.

Uses the [tirta123/noob-wiki](https://huggingface.co/datasets/tirta123/noob-wiki) dataset to automatically determine character franchises.

> [!WARNING]
> Character detection and franchise grouping may not always be correct. It is recommended to briefly check the generated folders after sorting.

Unrecognized or low-confidence characters are placed in the `unknown` folder.

> [!TIP]
> Use `settings.json` to customize character/franchise sorting and improve organization.
> See usage [here](#settingsjson)

## Requirements

- Python 3
- NVIDIA GPU for CUDA acceleration, **or**
- Compatible AMD GPU for ROCm/MIGraphX acceleration, **or**
- CPU

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv pyenv
```
2. Use the created virtual environment:

##### on Windows

```powershell
pyenv\Scripts\activate
```

##### on MacOS / Linux

```bash
source pyenv/bin/activate
```

3. Install the required packages for your setup:

Install base requirements, then install the extra requirements for your hardware. If you have no GPU, use the CPU-only requirements.

#### BASE Packages
```bash
pip install -r requirements.txt
```

#### NVIDIA GPU (CUDA)
```bash
pip install -r requirements-nvidia.txt
```
The exact CUDA/cuDNN requirements depend on the installed ONNX Runtime version and your NVIDIA driver.

#### AMD GPU / DirectML (Windows)
```bash
pip install onnxruntime-directml
```

#### AMD GPU / Rocm (Linux)
```bash
pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl numpy==1.26.4
```

#### CPU only
```bash
pip install -r requirements-cpu.txt
```

4. Usage

Place the images *(or folders with images)* you want to sort in the input folder and run the sorter:

```bash
python script.py
```

## settingsjson

Custom character grouping and overrides can be configured in:

```json
{
    "settings": {
        "CONFIDENCE": 0.80, // 1 is highest
        
        "CREATE_CHARACTER_FOLDER": true, // create folder for character?
        "CHARACTER_FOLDER_MIN_COUNT": 15, // min count of images to create the character name folder
        "REMOVE_EMPTY_FOLDERS": true, // remove empty folders in INPUT_PATH?

        "INPUT_PATH": "sorter/input", // input folder to process images from
        "UNKNOWN_PATH": "sorter/unknown", // not recognized images
        "OUTPUT_PATH": "sorter/output", // recognized and tagged images folder output
        "BACKUP_PATH": "sorter/backup" // path to backup images from INPUT_PATH
    },

    "overrides": {
        "sakura_miku": "hatsune_miku", // detected character name, overriden to character name
        "racing_miku": "hatsune_miku",
        "snow_miku": "hatsune_miku",
        "uruha_rushia_(3rd_costume)": "uruha_rushia",
        "boo_tao": "hu_tao"
    },

    "franchise_groups": {
        "BRS": [ // folder name
            "black_rock_shooter", // character name
            "dead_master"
        ],
        "Sousou_No_Frieren": [
            "frieren"
        ]
    }
}
```

`overrides` can be used to treat character variants as the same character. For example, `sakura_miku` and `racing_miku` can be sorted into the `Hatsune_Miku` folder.

`franchise_groups` can be used to manually group characters into a franchise. In this example, the sorter will create:

```text
BRS/Black_Rock_Shooter/

Sousou_No_Frieren/Frieren/
```

This is useful for characters such as `black_rock_shooter`, which may be detected by WD14 as `black_rock_shooter_(character)`, or `frieren`, which does not have a franchise tag in its WD14 name.
