#!/usr/bin/env python
import os

OD_RESEARCH_DIR = "/Users/illb/tensorflow/models/research"
TL_CLASSIFICATION_DIR = "/Users/illb/carnd/ROS-Racers-CarND-Capstone/tl_classification"

config_path = TL_CLASSIFICATION_DIR + "/config/ssd_mobilenet_v1_coco.config"
output_path = TL_CLASSIFICATION_DIR + "/output"

os.chdir(OD_RESEARCH_DIR)

cmd = "python object_detection/train.py"
cmd += " --logtostderr"
cmd += " --pipeline_config_path=" + config_path
cmd += " --train_dir=" + output_path

os.system(cmd)
