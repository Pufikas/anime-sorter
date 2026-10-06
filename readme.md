# anime-sorter

Detects anime characters from images and automatically sorts them into character and franchise folders.

Supported models:

| Model        | Speed  | VRAM   | Backend | Best for                   |
| ------------ | ------ | ------ | ------- | -------------------------- |
| `wd14`       | Fast   | Low    | ONNX    | Older/weaker hardware      |
| `pixai_onnx` | Medium | Medium | ONNX    | AMD DirectML / NVIDIA CUDA |
| `pixai`      | Slow   | High   | PyTorch | Best PixAI v1.0 experience |
| `pixai_onnx_fp32`      | Medium/Slow   | High   | ONNX | Best For AMD on Windows |

The `wd14` model uses the [tirta123/noob-wiki](https://huggingface.co/datasets/tirta123/noob-wiki) dataset to determine character franchises.

> [!WARNING]
> Character detection and franchise grouping may not always be correct. It is recommended to briefly check the generated folders after sorting.

Unrecognized or low-confidence characters are placed in the `unknown` folder.

> [!TIP]
> Use `settings.json` to customize character/franchise sorting and improve organization.
> See usage [here](#settingsjson)

## Requirements

- Python 3
- GPU acceleration is optional
- Enough RAM

Supported acceleration depends on the selected model and your hardware:

* NVIDIA GPU - CUDA *(best use case)*
* AMD GPU on Windows - **DirectML** for ONNX models
* AMD GPU on Linux - **ROCm/ONNX Runtime**, depending on compatibility
* CPU is also supported, but significantly slower for larger models


### Model requirements

#### `wd14`

Uses `imgutils` and ONNX Runtime.

This is the lightest model and is recommended for weaker hardware.

#### `pixai`

Uses PyTorch and Transformers.

This is the heaviest model and requires significantly more VRAM/RAM.

#### `pixai_onnx`

Uses the ONNX version of PixAI Tagger v0.9.

This avoids the PyTorch/Transformers requirements of the normal PixAI model and can use ONNX Runtime backends such as CUDA or DirectML.


## Setup

1. Create a Python virtual environment:

```bash
python -m venv pyenv
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
You will need to get [model_fp16.onnx](https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8)

> [!IMPORTANT]
> The normal `pixai` model uses PyTorch rather than ONNX Runtime. AMD Windows support therefore depends on PyTorch/DirectML compatibility and may require an older Python version.
>
> Python 3.12 is recommended if the normal `pixai` model cannot be installed correctly on newer Python versions.

#### AMD GPU / Rocm (Linux)

ONNX Runtime ROCm support depends on the ROCm and ONNX Runtime versions being compatible with your GPU and Linux installation.

#### CPU only
```bash
pip install -r requirements-cpu.txt
```

CPU inference works with all supported models, but heavier models will be a lot slower.

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
        "DEBUG": false,
        "MODEL": "wd14", // model to use supported are: ["pixai", "wd14", "pixai_onnx"], wd14 being the lightest and only having data up to 2024 while pixai is the heaviest with up to 2026 data

        "CONFIDENCE": 0.80, // 1 is highest, used for single character images
        "BATCH_SIZE": 6, // how many images to analyze per batch? more ==> more vram usage and faster processing
        "MULTI_CHARACTER_CONFIDENCE": 0.80, // used when determining whether multiple characters are present
        "MULTI_CHARACTER_MIN_COUNT": 2, // number of characters that must pass the multi-character confidence threshold before an image is treated as containing multiple characters

        "CREATE_CHARACTER_FOLDER": true, // create folder for character? if false moves the image into franchise folder only
        "CHARACTER_FOLDER_MIN_COUNT": 15, // min number of images required before creating a character folder

        "REMOVE_EMPTY_FOLDERS": true, // removes empty directories left in the input directory after sorting

        "CREATE_BACKUPS": false, // if enabled, images are copied to `BACKUP_PATH` before being moved.

        "INPUT_PATH": "sorter/input", // input folder to process images from
        "UNKNOWN_PATH": "sorter/unknown", // not recognized images
        "OUTPUT_PATH": "sorter/output", // recognized and tagged images folder output
        "BACKUP_PATH": "sorter/backup" // path to backup images from INPUT_PATH
    },

    "overrides": { 
        "boo_tao_(genshin_impact)": "hu_tao_(genshin_impact)" // detected character name, overriden to character name, currently must be with franchise tag
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
