import tensorflow as tf
import os
import numpy as np
import time

import rospy

from styx_msgs.msg import TrafficLight

class TLClassifier(object):

    @staticmethod
    def _load_graph(ckpt_path):
        rospy.loginfo('tl_classifier: _load_graph started')

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        rospy.loginfo('tl_classifier: _load_graph finished')
        return detection_graph


    def __init__(self):
        self.is_sim = True
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = current_dir + '/../model/faster_rcnn_inception_v2_coco_real'
        if self.is_sim:
            model_dir = current_dir + '/../model/faster_rcnn_inception_v2_coco_sim'
        ckpt_path = model_dir + '/frozen_inference_graph.pb'

        detection_graph = TLClassifier._load_graph(ckpt_path)
        self.detection_graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        time_start = time.time()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        elapsed = time.time() - time_start

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(scores) < 1 or scores[0] < 0.5:
            return TrafficLight.UNKNOWN

        cls = classes[0]
        rospy.loginfo("tl_classifier: class=" + cls + ", score=" + scores[0] + ", elapsed=" + str(elapsed))

        if cls == 1:
            return TrafficLight.RED
        elif cls == 2:
            return TrafficLight.YELLOW
        elif cls == 3:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
