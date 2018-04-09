#!/usr/bin/env python
import os
import glob
import re
import numpy as np

from config import *

os.chdir(OD_RESEARCH_DIR)

ckpt_files = glob.glob(OUTPUT_DIR + "/model.ckpt-*.index")
def ext_num(path):
    index = re.search('model\.ckpt-([0-9]+).index', path, re.IGNORECASE)
    if index:
        return int(index.group(1))
    else:
        return 0

max_index = max(map(ext_num, ckpt_files))
checkpoint_path = OUTPUT_DIR + "/model.ckpt-" + str(max_index)

cmd = "python object_detection/export_inference_graph.py"
cmd += " --logtostderr"
cmd += " --pipeline_config_path=" + CONFIG_PATH
cmd += " --trained_checkpoint_prefix=" + checkpoint_path
cmd += " --output_directory=" + OUTPUT_DIR

os.system(cmd)
