#!/usr/bin/env python
import os

from config import *

if not os.path.isdir(OD_RESEARCH_DIR):
    print("* check OD_RESEARCH_DIR in config.py")
    print("* check object detection API installation : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md")
    exit(-1)

if os.path.dirname(os.path.realpath(__file__)) != os.path.realpath(TL_CLASSIFICATION_DIR):
    print("* check TL_CLASSIFICATION_DIR in config.py")
    exit(-1)

model_pretrained_dir = TL_CLASSIFICATION_DIR + "/model_pretrained"
path, dirs, files = os.walk(model_pretrained_dir).next()
if len(dirs) <= 0:
    print("* check your model_pretrained directory : " + model_pretrained_dir)
    print("* download and extract pretrained models into model_pretrained directory : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")
    exit(-1)
