OD_RESEARCH_DIR = "/Users/illb/tensorflow/models/research"
TL_CLASSIFICATION_DIR = "/Users/illb/carnd/ROS-Racers-CarND-Capstone/tl_classification"

MODE = "sim" # sim | real
CONFIG_FILENAME = "ssd_mobilenet_v1_coco.config"

OUTPUT_DIR = TL_CLASSIFICATION_DIR + "/output_" + MODE

CONFIG_INPUT_PATH = TL_CLASSIFICATION_DIR + "/model/" + CONFIG_FILENAME
CONFIG_PATH = OUTPUT_DIR + "/" + CONFIG_FILENAME
EXPORT_CHECKPOINT_PATH = OUTPUT_DIR + "/model.ckpt-623"
