import json
from ocr_dataset import Label
from pydantic import ValidationError


split = 'train'
sample = "IMG_OCR_53_4PR_09180"

# Load Label from .json file
label_filename = f"data/{split}/{sample}.json"
with open(label_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

# parse the label using Pydantic
try:
    label = Label(**data)
    # print(label)
    # print(label.Annotation)
    # print(label.Dataset)
    # print(label.Dataset.category)
    # print(label.Dataset.identifier)
    print(label.Images)
    print('writer age:', label.Images.writer_age)
    print('writer sex:', label.Images.writer_sex)
    print(f'{len(label.bbox)} bbox')
except ValidationError as e:
    print("Validation errors:")
    print(e)

# load image
image_filename = f"data/{split}/{sample}.png"

from PIL import Image
Image.open(image_filename).show()