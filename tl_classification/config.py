OD_RESEARCH_DIR = "/Users/illb/tensorflow/models/research"
TL_CLASSIFICATION_DIR = "/Users/illb/carnd/ROS-Racers-CarND-Capstone/tl_classification"

MODE = "sim" # sim | real
MODEL_NAME = "ssd_mobilenet_v1_coco"

OUTPUT_DIR = TL_CLASSIFICATION_DIR + "/output_" + MODE

CONFIG_INPUT_PATH = TL_CLASSIFICATION_DIR + "/model_config/" + MODEL_NAME + ".config"
CONFIG_PATH = OUTPUT_DIR + "/" + MODEL_NAME + ".config"
EXPORT_CHECKPOINT_PATH = OUTPUT_DIR + "/model.ckpt-623"
