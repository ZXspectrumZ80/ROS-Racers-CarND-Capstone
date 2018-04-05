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

