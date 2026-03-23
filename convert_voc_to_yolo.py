"""
Convert Pascal VOC XML annotations to YOLO format.

Dataset: NEU Surface Defect Dataset
"""

import os
import xml.etree.ElementTree as ET

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

xml_dir = r"dataset/validation/annotations"
img_dir = r"dataset/validation/images"
label_dir = r"dataset/validation/labels"

os.makedirs(label_dir, exist_ok=True)


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x *= dw
    w *= dw
    y *= dh
    h *= dh

    return x, y, w, h


for file in os.listdir(xml_dir):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_file = open(os.path.join(label_dir, file.replace(".xml", ".txt")), "w")

    for obj in root.iter("object"):
        cls = obj.find("name").text
        cls_id = classes.index(cls)

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )

        bb = convert((w, h), b)
        label_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

    label_file.close()
