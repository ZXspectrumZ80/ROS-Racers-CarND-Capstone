OD_RESEARCH_DIR = "/Users/illb/tensorflow/models/research"
TL_CLASSIFICATION_DIR = "/Users/illb/carnd/ROS-Racers-CarND-Capstone/tl_classification"
OUTPUT_DIR = TL_CLASSIFICATION_DIR + "/output"

MODE = "sim" # sim | real
CONFIG_PATH = TL_CLASSIFICATION_DIR + "/config/ssd_mobilenet_v1_" + MODE + ".config"
EXPORT_CHECKPOINT_PATH = OUTPUT_DIR + "/model.ckpt-623"
