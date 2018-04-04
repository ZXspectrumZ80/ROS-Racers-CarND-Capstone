#!/usr/bin/env python

try:
    import xmltodict
except ImportError:
    raise ImportError('pip install xmltodict')

import json
import glob
import os

from config import *


def convert(postfix, files):
    images = []
    for file in files:
        with open(file, "rb") as f:
            x = xmltodict.parse(f)
            try:
                a = x["annotation"]
                o_list = [a["object"]]
                if type(a["object"]) is list:
                    o_list = a["object"]
                objs = []
                for o in o_list:
                    bbox = o["bndbox"]
                    objs.append({
                        "class": o["name"],
                        "bbox": [int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])]
                    })
                image = {
                    "folder": a["folder"],
                    "filename": a["filename"],
                    "objects": objs
                }
                images.append(image)
            except:
                print("convert error:" + file)


    with open("data/annotations_" + postfix + ".json", 'w') as f:
        f.write(json.dumps({"images":images}, indent=2))


for s in ["train", "val"]:
    postfix = MODE
    if s == "val":
        postfix += "_val"
    files = glob.glob("data/images_" + postfix + '/*.xml')
    convert(postfix, files)

