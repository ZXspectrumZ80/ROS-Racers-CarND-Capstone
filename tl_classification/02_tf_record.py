#!/usr/bin/env python

import tensorflow as tf
import json
import os

from object_detection.utils import dataset_util
from config import *

LABEL_DICT =  {
  "red" : 1,
  "yellow" : 2,
  "green" : 3
}

def create_tf_example(example):
  height = 600 # Image height
  width = 800 # Image width

  filename = example['filename'].encode() # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_path = "data/" + example['folder'] + "/" + example['filename']
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_image_data = fid.read()

  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  try:
    for obj in example['objects']:
      bbox = obj["bbox"]
      xmins.append(float(bbox[0] * 1.0/ width))
      xmaxs.append(float(bbox[2] * 1.0/ width))
      ymins.append(float(bbox[1] * 1.0/ height))
      ymaxs.append(float(bbox[3] * 1.0/ height))
      classes_text.append(obj['class'].encode())
      classes.append(int(LABEL_DICT[obj['class']]))
  except TypeError as e:
    print("type error:" + image_path)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def record(postfix):
  output_path = OUTPUT_DIR + "/tl_" + postfix + ".record"
  writer = tf.python_io.TFRecordWriter(output_path)

  annotations_file = OUTPUT_DIR + "/annotations_" + postfix + ".json"

  images = []
  with tf.gfile.GFile(annotations_file, 'r') as fid:
    groundtruth_data = json.load(fid)
    images = groundtruth_data["images"]

  for example in images:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


def main(_):
  for s in ["train", "val"]:
    record(s)


if __name__ == '__main__':
  tf.app.run()