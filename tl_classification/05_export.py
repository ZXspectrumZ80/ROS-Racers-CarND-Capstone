#!/usr/bin/env python
import os

from config import *

os.chdir(OD_RESEARCH_DIR)

cmd = "python object_detection/export_inference_graph.py"
cmd += " --logtostderr"
cmd += " --pipeline_config_path=" + CONFIG_PATH
cmd += " --trained_checkpoint_prefix=" + EXPORT_CHECKPOINT_PATH
cmd += " --output_directory=" + OUTPUT_DIR

os.system(cmd)
