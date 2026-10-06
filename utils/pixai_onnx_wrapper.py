class PixAIOnnxTagger:
    def __init__(self, session, tags):
        self.session = session
        self.tags = tags

        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, image, batch_size=None):
        from PIL import Image
        import numpy as np

        # since we send a path to image (instead of raw image)
        # open with PIL instead
        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")
        image = self._preprocess(image) # converts the PIL image to format expected by the pixai onnx model

        # [0](1) => gets output array from onnx runtime
        # [0](2) => gets the first image in the batch
        logits = self.session.run(
            [self.output_name],
            { self.input_name: image }
        )[0][0]

        probabilities = 1 / (1 + np.exp(-logits))

        character = [] # char names
        copyright = [] # franchise names

        for i, tag in self.tags["character"]:
            probability = float(probabilities[i])

            if probability >= 0.5:
                character.append({
                    "tag": tag,
                    "confidence": probability
                })

        for index, tag in self.tags["copyright"]:
            probability = float(probabilities[index])

            if probability >= 0.5:
                copyright.append({
                    "tag": tag,
                    "confidence": probability
                })

        return {
            "characters": character,
            "franchises": copyright
        }

    def _preprocess(self, image):
        import numpy as np
        from PIL import Image

        target = 1008 # pixai expects 1008x1008 inputs
        width, height = image.size

        scale = min(target / width, target / height) # scale the image without stretching it

        new_width = round(width * scale)
        new_height = round(height * scale)

        image = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )

        # create the resized canvas, empty areas are replaced with black color
        canvas = Image.new("RGB", (target, target), (0, 0, 0))

        # centering the image
        x = (target - new_width) // 2
        y = (target - new_height) // 2

        canvas.paste(image, (x, y))

        # PIL image to numpy array convertion
        array = np.asarray(canvas, dtype = np.float32)

        array = array / 255.0 # 0-255 => 0-1
        array = array * 2.0 - 1.0 # 0-1 => -1 to 1 (needed for pixai)

        array = np.transpose(array, (2, 0, 1)) # for onnx model

        # batch dimensions
        # [3, 1008, 1008] => [1, 3, 1008, 1008]
        array = np.expand_dims(array, axis = 0)

        return array

def load_tags_json(tags_path):
    import json

    with open(tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = {
        category["name"]: category
        for category in data["categories"]
    }

    return {
        "character": [
            (tags["character"]["offset"] + i, tag)
            for i, tag in enumerate(tags["character"]["tags"])
        ],

        "copyright": [
            (tags["copyright"]["offset"] + i, tag)
            for i, tag in enumerate(tags["copyright"]["tags"])
        ]
    }

def load_tags_csv(tags_path):
    import csv

    tags = {
        "character": [],
        "copyright": []
    }

    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader) # skips first line (header)

        for index, row in enumerate(reader):
            tag = row[1]
            category = int(row[2])

            if category == 4:
                tags["character"].append((index, tag))

            elif category == 3:
                tags["copyright"].append((index, tag))

    return tags