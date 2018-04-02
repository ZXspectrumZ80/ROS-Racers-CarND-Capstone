#!/usr/bin/env python
import os

OD_RESEARCH_DIR = "/Users/illb/tensorflow/models/research"
TL_CLASSIFICATION_DIR = "/Users/illb/carnd/ROS-Racers-CarND-Capstone/tl_classification"

config_path = TL_CLASSIFICATION_DIR + "/config/ssd_mobilenet_v1_coco.config"
output_path = TL_CLASSIFICATION_DIR + "/output"

os.chdir(OD_RESEARCH_DIR)

job = "train"
#job = "export"
cmd = ""

if job == "train":
	cmd = "python object_detection/train.py"
	cmd += " --logtostderr"
	cmd += " --pipeline_config_path=" + config_path
	cmd += " --train_dir=" + output_path
elif job == "export":
	checkpoint_path = output_path + "/model.ckpt-623"
	cmd = "python object_detection/export_inference_graph.py"
	cmd += " --logtostderr"
	cmd += " --pipeline_config_path=" + config_path
	cmd += " --trained_checkpoint_prefix=" + checkpoint_path
	cmd += " --output_directory=" + output_path

os.system(cmd)
