#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from math import sqrt

STATE_COUNT_THRESHOLD = 3       # repetitive traffic light state to take action
SLOW_DOWN_DISTANCE    = 120     # The distance from the red traffic light to start to slow down

# when the Flag is set the planner gets the traffic data directly from the Simulator
SIMULATOR_TRAFFIC_ENABLED = True      # Toggle True or False

PUBLISH_RATE = 10               # 10Hz - loop rate

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.has_image                = None
        self.egoCar_pose              = None
        self.full_track_wpts          = None
        self.front_camera_image       = None
        self.traffic_lights_List      = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.egoCar_pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.track_waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_lights_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.camera_image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config   = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge                      = CvBridge()
        self.light_classifier            = TLClassifier()
        self.listener                    = tf.TransformListener()

        self.light_state                 = TrafficLight.UNKNOWN
        self.previous_light_state        = TrafficLight.UNKNOWN
        self.previous_light_wp           = -1
        self.red_light_state_count       = 0
        self.light_wp_count              = 0

        self.closest_TL_EgoCar_distance  = float("inf")
        self.previous_TL_EgoCar_distance = 0.0
        self.closest_TL_index            = -1
        self.stopline_wp_index           = -1

        # run process to update Traffic light status and stop lines
        self.loop()

    ###########################################################################

    def loop(self):
        """ 
                loop() - continuous loop that processes messages and publishes traffic light stop lines
        """
        Iteration_rate = rospy.Rate(PUBLISH_RATE)  # 10 Hz  / previously was 2 Hz

        while not rospy.is_shutdown():
            if (self.egoCar_pose is not None) and (self.full_track_wpts is not None):
                #
                self.process_traffic_lights()
                #
                #self.publish_traffic_light()
                # pause for delay time
                Iteration_rate.sleep()

    ###########################################################################

    def egoCar_pose_cb(self, msg):
        self.egoCar_pose = msg.pose

    ###########################################################################

    def track_waypoints_cb(self, waypoints):
        self.full_track_wpts  = waypoints

    ###########################################################################

    def traffic_lights_cb(self, msg):
        self.traffic_lights_List = msg.lights

    ###########################################################################

    def camera_image_cb(self, msg):
        self.has_image = True
        self.front_camera_image = msg

    ###########################################################################

    def publish_traffic_light(self):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if (self.light_state == self.previous_light_state) and (self.light_state == TrafficLight.RED):
            self.red_light_state_count += 1
            self.previous_light_state = self.light_state
        else:
            self.red_light_state_count = 0
            self.previous_light_state  = self.light_state

        if self.red_light_state_count >= STATE_COUNT_THRESHOLD:
            self.upcoming_red_light_pub.publish(Int32(self.stopline_wp_index))
        else:
            self.upcoming_red_light_pub.publish(Int32(-1))

    ###########################################################################

    def distance(self, p1, p2):
        xs = pow((p1.position.x - p2.position.x), 2)
        ys = pow((p1.position.y - p2.position.y), 2)
        return sqrt(xs + ys)

    ###########################################################################

    def dist(self, p1, p2):
        """
        dist()
        calculates the euclidean distance between two points
        Input: 
            p1 - first point
            p2 - second point
        Return:
        		dist - distance between the two input points
        """
        return sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2)

    ###########################################################################

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        min_dist = float("inf")
        closest_wp_idx = -1

        if not waypoints:
            rospy.logwarn("no waypoints")
        else:
            for idx, wp in enumerate(waypoints):
                dist = self.distance(pose, wp.pose.pose)
                if (dist < min_dist):
                    min_dist = dist
                    closest_wp_idx = idx
        return closest_wp_idx

    ###########################################################################

    def get_traffic_light_state_from_camera(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.front_camera_image, "bgr8")

        #Get classification
        self.light_state = self.light_classifier.get_classification(cv_image)

    ###########################################################################

    def get_traffic_lights(self):
        """  
            Determines the closet traffic light position from the ego car and it color status

                Args:
                    TLs       : traffic light list
                    egoCar_pos: current position of the ego car
                    track waypoints: full track waypoints

                Returns:
                    int  : ID of traffic light color (specified in styx_msgs/TrafficLight)
                    index: the index of the full track waypoint at which the traffic light is located.

        """

        TLs = self.traffic_lights_List
        number_of_traffic_lights = len(TLs)
        # print(" number_of_traffic_lights %d" % number_of_traffic_lights)

        closest_TL_index = -1
        TL_state         = TrafficLight.UNKNOWN

        closest_TL_EgoCar_distance = float("inf")  # Any big number

        for i in range(0,number_of_traffic_lights):

            # compares each traffic Light pose to ego Car pose if Light is RED
            TL_EgoCar_distance = self.dist(self.egoCar_pose.position, TLs[i].pose.pose.position)

            #if (TL_EgoCar_distance < closest_TL_EgoCar_distance) and (TLs[i].state == TrafficLight.RED):
            if (TL_EgoCar_distance < closest_TL_EgoCar_distance):
                closest_TL_EgoCar_distance = TL_EgoCar_distance
                closest_TL_index           = i
                TL_state                   = TLs[closest_TL_index].state

        self.closest_TL_EgoCar_distance = closest_TL_EgoCar_distance
        self.closest_TL_index           = closest_TL_index
        self.light_state                = TL_state

        #print(" closest_TL_index %d" % self.closest_TL_index)
        #print(" closest_TL_EgoCar_distance %f" % self.closest_TL_EgoCar_distance)
        #print(" TL_state %d" % TL_state)


        if (closest_TL_index == -1):
            self.stopline_wp_index = -1
            TL_state = TrafficLight.UNKNOWN
            self.light_state = TL_state
            return

        if (self.closest_TL_EgoCar_distance > SLOW_DOWN_DISTANCE):
            self.stopline_wp_index = -1
            self.light_state = TrafficLight.UNKNOWN
            self.previous_TL_EgoCar_distance = self.closest_TL_EgoCar_distance
            return

        closest_TL_wpts_distance = float("inf")
        closest_TL_wp_index      = -1

        # compares each track waypoint to the traffic light pose
        for i in range(len(self.full_track_wpts.waypoints)):
            wpt_TL_distance = self.dist(TLs[self.closest_TL_index].pose.pose.position,
                                        self.full_track_wpts.waypoints[i].pose.pose.position)
            if wpt_TL_distance < closest_TL_wpts_distance:
                closest_TL_wpts_distance = wpt_TL_distance
                closest_TL_wp_index = i


        # Check if the closest traffic light is ahead or behind the Ego Car
        if ((self.closest_TL_EgoCar_distance - self.previous_TL_EgoCar_distance) < 0):
            self.stopline_wp_index = closest_TL_wp_index
            TL_state               = TLs[self.closest_TL_index].state
        else:
            self.stopline_wp_index = -1
            TL_state               = TrafficLight.UNKNOWN

        self.previous_TL_EgoCar_distance = self.closest_TL_EgoCar_distance

        #print(" self.stopline_wp_index %d" % self.stopline_wp_index)
        #print(" self.egoCar_closest_wp_index %d" % self.egoCar_closest_wp_index)

        self.light_state = TL_state

    ###########################################################################

    def process_traffic_lights(self):

        if (SIMULATOR_TRAFFIC_ENABLED and self.egoCar_pose is not None and self.traffic_lights_List is not None):

            # Get the traffic light waypoint index and color status
            self.get_traffic_lights()
            self.publish_traffic_light()

        elif (self.egoCar_pose is not None and self.traffic_lights_List is not None):

            # Get the traffic light waypoint index
            self.get_traffic_lights()

            # Get the traffic light status from the classifier not from the simulator
            if self.stopline_wp_index > 0:
                self.light_state = self.get_traffic_light_state_from_camera()

            self.publish_traffic_light()

    ###########################################################################

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
