# anime-sorter

Detects anime characters from images and sorts them into folders.

> [!WARNING]
> Character detection and franchise grouping may not always be correct. It is recommended to briefly check the generated folders after sorting.

Unrecognized or low-confidence characters are placed in the `unknown` folder.

> [!TIP]
> Use `franchises.json` to customize character/franchise sorting and improve organization.

## Requirements

* Python 3
* NVIDIA GPU with CUDA support
* `onnxruntime-gpu`

## Setup

Create a Python virtual environment:

```bash
python3 -m venv pyenv
```

### Windows

```powershell
pyenv\Scripts\activate
```

### macOS / Linux

```bash
source pyenv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Place the images you want to sort in the input folder and run the sorter:

```bash
python script.py
```

## Customizing franchises

The sorter uses the [tirta123/noob-wiki](https://huggingface.co/datasets/tirta123/noob-wiki) dataset to automatically determine character franchises.

Custom character grouping and overrides can be configured in:

```text
franchises.json
```

For example:

```json
{
    "overrides": {
        "sakura_miku": "hatsune_miku",
        "racing_miku": "hatsune_miku"
    },

    "franchise_groups": {
        "BRS": [
            "black_rock_shooter"
        ],
        "sousou_no_frieren": [
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

> [!TIP]
> It is recommended to update `franchises.json` periodically if you encounter characters that are sorted incorrectly.

## GPU

Uses `onnxruntime-gpu` for GPU-accelerated image tagging.

The exact CUDA/cuDNN requirements depend on the installed ONNX Runtime version and your NVIDIA driver.

It is also possible to run this project with `onnxruntime` using only the CPU, but the sorting process will be much slower.
