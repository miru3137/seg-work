#/bin/python3
#
# This file is part of https://github.com/martinruenz/maskfusion
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import os

# Directories
PYTHON_VE_PATH = ""
MASK_RCNN_DIR = "/home/ashrat139/Workspace/seg-work/dep/Mask_RCNN"

# Optionally, activate virtual environment
if PYTHON_VE_PATH != "":
  ve_path = os.path.join(PYTHON_VE_PATH, 'bin', 'activate_this.py')
  exec(open(ve_path).read(), {'__file__': ve_path})

## Optionally, select GPU # this approach did not work
#if "@MASKFUSION_GPUS_MASKRCNN@" != "":
#  os.environ["CUDA_VISIBLE_DEVICES"] = "@MASKFUSION_GPUS_MASKRCNN@"
#  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, MASK_RCNN_DIR)

import random
import math
import numpy as np
from PIL import Image
import time
import scipy.misc
import pytoml as toml

import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # if "@MASKFUSION_GPUS_MASKRCNN@" != "":
# #     config.gpu_options.visible_device_list="@MASKFUSION_GPUS_MASKRCNN@"
# session = tf.Session(config=config)
# from keras.backend.tensorflow_backend import set_session, clear_session, _SESSION
# set_session(session)

from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from helpers import *

# Global variables (used to communicate with c++)
current_segmentation = None
current_class_ids = None
current_bounding_boxes = None

# Root directory of the project
DATA_DIR = os.path.join(MASK_RCNN_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "logs")

# PARAMETERS

# This only has an effect if the list contains at least 1 class name:
SPECIAL_ASSIGNMENTS = {}  # Example: {'person': 255}
SINGLE_INSTANCES = False
OUTPUT_FRAMES = True
STORE_CLASS_IDS = True
START_INDEX = 0

with open("config.toml", 'rb') as toml_file:
  toml_config = toml.load(toml_file)
  class_names = toml_config["MaskRCNN"]["class_names"]
  model_path = toml_config["MaskRCNN"]["model_path"]
  filter_classes = toml_config["MaskRCNN"]["filter_classes"]
  score_threshold = toml_config["MaskRCNN"]["score_threshold"]

class InferenceConfig(coco.CocoConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  NUM_CLASSES = len(class_names)

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(model_path, by_name=True)

filter_classes = [class_names.index(x) for x in filter_classes]
SPECIAL_ASSIGNMENTS = {class_names.index(x): SPECIAL_ASSIGNMENTS[x] for x in SPECIAL_ASSIGNMENTS}

def execute(rgb_image):
  global current_segmentation
  global current_class_ids
  global current_bounding_boxes
  results = model.detect([rgb_image], verbose=0)
  r = results[0]
  current_segmentation, current_class_ids, current_bounding_boxes = generate_id_image(r, score_threshold, filter_classes, SPECIAL_ASSIGNMENTS)
