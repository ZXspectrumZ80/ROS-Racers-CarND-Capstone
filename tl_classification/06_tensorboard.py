#!/usr/bin/env python
import os

from config import *

logdir = OUTPUT_DIR

cmd = "tensorboard --logdir=" + logdir

os.system(cmd)
