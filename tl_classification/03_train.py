#!/usr/bin/env python
import os

from config import *

os.chdir(OD_RESEARCH_DIR)

cmd = "python object_detection/train.py"
cmd += " --logtostderr"
cmd += " --pipeline_config_path=" + CONFIG_PATH
cmd += " --train_dir=" + OUTPUT_DIR

os.system(cmd)
