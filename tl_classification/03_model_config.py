#!/usr/bin/env python

from config import *

def create_model_config():
    lines = []
    with open(CONFIG_INPUT_PATH, "r") as f:
        lines = f.readlines()

    def _replace(x):
        x = x.replace("${TL_CLASSIFICATION_DIR}", TL_CLASSIFICATION_DIR)
        if MODE == "real":
            x = x.replace("${EVAL_NUM_EXAMPLES}", str(3))
        else:
            x = x.replace("${EVAL_NUM_EXAMPLES}", str(7))
        x = x.replace("${MODE}", MODE)
        return x

    lines = map(_replace, lines)

    with open(CONFIG_PATH, 'w') as f:
        f.writelines(lines)


create_model_config()