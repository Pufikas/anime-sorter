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

Character/franchise mappings can be configured in:

```text
franchises.json
```

This allows characters to be grouped into franchise folders, if they are missing tags from `wd14` library, or to keep the folders organized.

For example:

```json
{
    "BRS": [
        "black_rock_shooter"
    ],

    "sousou_no_frieren": [
        "frieren"
    ],
}
```

In this example this will create `BRS/Black_Rock_Shooter` and `Sousou_No_Frieren/Frieren` folders.

For example, `black_rock_shooter` may be detected by `WD14` as `black_rock_shooter_(character)`. Without a custom mapping, this would result in a `character` folder being created instead of `BRS`.

Some characters also do not have a franchise tag in their `WD14` name. For example, `frieren` does not contain the `sousou_no_frieren` franchise tag, while `fern` does. Adding `frieren` to `franchises.json` allows it to be grouped correctly.

> [!TIP]
> It is recommended to update `franchises.json` periodically 


## GPU

Uses `onnxruntime-gpu` for GPU-accelerated image tagging.

The exact CUDA/cuDNN requirements depend on the installed ONNX Runtime version and your NVIDIA driver.

It is also possible to run this project with `onxruntime` using only CPU, but will result in much slower process.
