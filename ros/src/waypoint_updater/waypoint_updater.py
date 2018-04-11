#!/usr/bin/env python

import os
import sys

import math
import rospy
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS    = 200 	 # Number of waypoints we will publish. You can change this number
MAX_DECELERATION = 0.5
RED_LIGHT        = 0
BIG_NUMBER       = 1e5   # to be used for search routines
PUBLISH_RATE     = 10    # 20 Hz

# to stop before the traffic light (on the stop line) - highly dependant on the target speed
STOP_MARGIN      = 28.0  # Stop distance before the traffic light (5.0 meters in test lot)
Kp_SPEED         = 0.35  # Proportional Controller Coeff for EgoCar speed regulation
#TARGET_SPEED_MPH = 30   # Miles per Hour
#TARGET_SPEED_MPS = (TARGET_SPEED_MPH * 1609) / (60 * 60)    # Meter per Sec

# when the Flag is set the planner gets the traffic data directly from the Simulator
SIMULATOR_TRAFFIC_ENABLED = True



class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')        # log_level=rospy.DEBUG can be inserted later

        # subscriber nodes
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.trafficLights_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)

        # publisher nodes
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # member variables
        self.egoCar_pose                 = None
        self.egoCar_heading              = None
        self.egoCar_velocity             = 0.0
        self.full_track_wpts             = None
        self.traffic_waypoint            = None
        self.traffic_lights_List         = None

        self.stopline_wp_index           = -1
        self.egoCar_closest_wp_index     = None
        self.egoCar_ahead_waypoints      = None
        self.egoCar_lane                 = None
        self.closest_TL_EgoCar_distance  = BIG_NUMBER
        self.previous_TL_EgoCar_distance = 0.0
        self.closest_TL_index            = -1

        # get the max speed limit from the "waypoint_loader" launch file
        # Convert from KM/hour to Meter/sec
        self.TARGET_SPEED_MpS = self.KMpH_to_MpS(rospy.get_param('~/waypoint_loader/velocity'))

        # run process to update EgoCar waypoints
        self.loop()
    ###########################################################################

    def loop(self):
        """ 
        loop() - continuous loop that processes messages and publishes final waypoints
        """
        Update_rate = rospy.Rate(PUBLISH_RATE)        # 10 Hz  / previously was 2 Hz
        while not rospy.is_shutdown():
            if (self.egoCar_pose is not None) and (self.full_track_wpts is not None):
                # get index of next waypoint
                self.get_egoCar_first_wp_index()
                # generate final_waypoints
                self.get_egoCar_ahead_waypoints()
                # publish message
                self.publish_egoCar_waypoints()
                # pause for delay time
                Update_rate.sleep()
    ############################################################################

    def get_closest_track_waypoint(self):
        """ 
        get_closest_track_waypoint(pose)
        gets closest waypoint index, leveraged from path planning
        Input: 
            current pose
        Return: 
            index of closest waypoint
        """
        closest_wp_distance = BIG_NUMBER        # Any big number

        # sets up pose and waypoints
        if self.full_track_wpts is not None:
            full_track_wpts = self.full_track_wpts.waypoints
        else:
            return -1

        # compares each waypoint to current pose
        for i in range(len(full_track_wpts)):
            wpt_pose_distance = self.dist(self.egoCar_pose.position,
                                          full_track_wpts[i].pose.pose.position)
            if wpt_pose_distance < closest_wp_distance:
                closest_wp_distance = wpt_pose_distance
                closest_track_wp_index = i

        # Check if the closest point is ahead or behind the Ego Car
        closest_track_wp_vector = np.array([full_track_wpts[closest_track_wp_index].pose.pose.position.x ,
                                            full_track_wpts[closest_track_wp_index].pose.pose.position.y])
        prev_track_wp_vector    = np.array([full_track_wpts[closest_track_wp_index-2].pose.pose.position.x ,
                                            full_track_wpts[closest_track_wp_index-2].pose.pose.position.y])
        egoCar_pos_vector       = np.array([self.egoCar_pose.position.x,
                                            self.egoCar_pose.position.y])

        DOT_Product_val = np.dot( (closest_track_wp_vector - prev_track_wp_vector),
                                  (egoCar_pos_vector - closest_track_wp_vector) )

        if DOT_Product_val > 0:      # The Closest Point is behind the Ego Car
            closest_track_wp_index = (closest_track_wp_index + 2) % len(full_track_wpts)
            # increment the index by 2 step (by trial and error)

        return closest_track_wp_index
    ###########################################################################

    def get_egoCar_first_wp_index(self):
        """ 
        get_next_waypoint(pose)
        gets next waypoint index, leveraged from path planning
        Input: 
            current pose
        Return: 
            index of next waypoint
        """
        # gets closest waypoint

        egoCar_first_wp_index = self.get_closest_track_waypoint()
        p1 = self.egoCar_pose.position
        p2 = self.full_track_wpts.waypoints[egoCar_first_wp_index].pose.pose.position

        # checks angle and increases index if angle too large
        heading = math.atan2((p2.y - p1.y), (p2.x - p1.x))
        if self.egoCar_heading is not None:
            angle = abs(self.egoCar_heading - heading)
            if angle > math.pi/4:
                egoCar_first_wp_index += 1
        self.egoCar_closest_wp_index = egoCar_first_wp_index

    ###########################################################################

    def get_egoCar_ahead_waypoints(self):
        """ 
        get_egoCar_ahead_waypoints()
        gets the waypoints from the index to the number of look ahead waypoints
        Input: 
            index in waypoint list
        Output: 
            list of waypoints to publish
        """
        # setup list

        full_track_length = len(self.full_track_wpts.waypoints)
        egoCar_ahead_waypoints = []

        idx  = self.egoCar_closest_wp_index

        for i in range(LOOKAHEAD_WPS):
            egoCar_ahead_waypoints.append(self.full_track_wpts.waypoints[(idx + i)% full_track_length])

        #print(" egoCar_ahead_waypoints size:", len(egoCar_ahead_waypoints))

        self.egoCar_ahead_waypoints = egoCar_ahead_waypoints

    ###########################################################################

    def publish_egoCar_waypoints(self):
        """
        publish_waypoints(waypoints)
        sets up the header and publishes the waypoint message
        this is used by the simulator to drive forward at the specified speed.
        Input: 
            waypoints - final set of waypoints
        """

        if ((self.traffic_lights_List is not None) and SIMULATOR_TRAFFIC_ENABLED):
            self.get_RED_traffic_lights()
        # setup list of waypoints
        self.generate_egoCar_lane()
        # publish message
        self.final_waypoints_pub.publish(self.egoCar_lane)

    ###########################################################################

    def generate_egoCar_lane(self):

        egoCar_lane = Lane()
        egoCar_lane.header.frame_id = '/world'
        egoCar_lane.header.stamp = rospy.Time(0)

        farthest_egoCar_wp_index = (self.egoCar_closest_wp_index + LOOKAHEAD_WPS)

        if (self.stopline_wp_index == -1) or (self.stopline_wp_index >= farthest_egoCar_wp_index):
            egoCar_lane.waypoints = self.egoCar_ahead_waypoints
            for i in range(0, len(egoCar_lane.waypoints)):
                # EgoCar speed regulator (for testing)
                desired_waypoint_speed = self.egoCar_velocity + \
                                         Kp_SPEED * (self.TARGET_SPEED_MpS - self.egoCar_velocity)
                #desired_waypoint_speed = self.TARGET_SPEED_MpS
                self.set_waypoint_velocity(egoCar_lane.waypoints, i  , desired_waypoint_speed)
        else:
            egoCar_lane.waypoints = self.decelerate_egoCar_waypoints()

        self.egoCar_lane = egoCar_lane

    ###########################################################################

    def decelerate_egoCar_waypoints(self):

        Updated_egoCar_ahead_waypoints = []
        EgoCar_wpts_length = len(self.egoCar_ahead_waypoints)
        # TL_stop_wp = self.full_track_wpts.waypoints[self.stopline_wp_index].pose.pose.position

        for i in range(0,EgoCar_wpts_length):

            ahead_wp = self.egoCar_ahead_waypoints[i]
            # ahead_wp_position = ahead_wp.pose.pose.position

            # egoCar_wp_TL_stop_dist = self.dist(ahead_wp_position, TL_stop_wp)
            egoCar_wp_TL_stop_dist = self.distance(self.full_track_wpts.waypoints,
                                                   self.egoCar_closest_wp_index + i, self.stopline_wp_index)
            EgoCar_velocity = math.sqrt(2 * MAX_DECELERATION * max(0, egoCar_wp_TL_stop_dist - STOP_MARGIN))
            if EgoCar_velocity < 1.0:
                EgoCar_velocity = 0.0
            ahead_wp.twist.twist.linear.x = min(EgoCar_velocity, ahead_wp.twist.twist.linear.x)
            Updated_egoCar_ahead_waypoints.append(ahead_wp)

        return Updated_egoCar_ahead_waypoints

    ###########################################################################

    def get_RED_traffic_lights(self):

        TLs = self.traffic_lights_List.lights
        number_of_traffic_lights = len(TLs)
        #print(" number_of_traffic_lights %d" % number_of_traffic_lights)

        closest_TL_index = -1

        closest_TL_EgoCar_distance = BIG_NUMBER  # Any big number

        for i in range(0,number_of_traffic_lights):

            # TL_status = TLs[i].state

            # compares each traffic Light pose to ego Car pose if Light is RED
            TL_EgoCar_distance = self.dist(self.egoCar_pose.position, TLs[i].pose.pose.position)

            if (TL_EgoCar_distance < closest_TL_EgoCar_distance) and (TLs[i].state == RED_LIGHT):
                closest_TL_EgoCar_distance = TL_EgoCar_distance
                closest_TL_index = i

        self.closest_TL_EgoCar_distance = closest_TL_EgoCar_distance
        self.closest_TL_index           = closest_TL_index

        #print(" closest_TL_index %d" % self.closest_TL_index)
        #print(" closest_TL_EgoCar_distance %f" % self.closest_TL_EgoCar_distance)

        if self.full_track_wpts is not None:
            full_track_wpts = self.full_track_wpts.waypoints
        else:
            return -1

        closest_TL_wpts_distance = BIG_NUMBER

        # compares each track waypoint to the traffic light pose
        for i in range(len(full_track_wpts)):
            wpt_TL_distance = self.dist(TLs[self.closest_TL_index].pose.pose.position,
                                        full_track_wpts[i].pose.pose.position)
            if wpt_TL_distance < closest_TL_wpts_distance:
                closest_TL_wpts_distance = wpt_TL_distance
                closest_TL_wp_index = i


        # Check if the closest traffic light  is ahead or behind the Ego Car
        if ((self.closest_TL_EgoCar_distance - self.previous_TL_EgoCar_distance) < 0):
            self.stopline_wp_index = closest_TL_wp_index
        else:
            self.stopline_wp_index = -1

        self.previous_TL_EgoCar_distance = self.closest_TL_EgoCar_distance

        #print(" self.stopline_wp_index %d" % self.stopline_wp_index)
        #print(" self.egoCar_closest_wp_index %d" % self.egoCar_closest_wp_index)

    ###########################################################################

    def pose_cb(self, msg):
        """
        callback function that sets the current pose of car (position/orientation)
        Input: 
            ROS PoseStamped msg
        """
        self.egoCar_pose = msg.pose
        # calculate the quaternion to get heading
        quaternion = (self.egoCar_pose.orientation.x,
                      self.egoCar_pose.orientation.y,
                      self.egoCar_pose.orientation.z,
                      self.egoCar_pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.egoCar_heading = yaw

    ###########################################################################

    def waypoints_cb(self, waypoints):
        """
        callback function that sets the base waypoints of the capstone track
        Input: 
            list of waypoints
        """
        self.full_track_wpts = waypoints

    ###########################################################################

    def traffic_cb(self, msg):
        """
        callback function that sets the waypoints of the capstone track traffic
        Input: 
            list of traffic waypoints
        """
        #self.traffic_waypoint = msg.data
        self.stopline_wp_index = msg.data

    ###########################################################################

    def trafficLights_cb(self, msg):
        """
        callback function that sets the waypoints of the capstone track traffic
        Input: 
            list of traffic waypoints
        """
        #self.traffic_waypoint = msg.data
        self.traffic_lights_List = msg

    ###########################################################################
            
    def velocity_cb(self, msg):
        """
        callback function for the twist message and sets the current velocity
        Input: 
            twist message
        """
        self.egoCar_velocity = msg.twist.linear.x

    ###########################################################################

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    ###########################################################################

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x
    ###########################################################################

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity
    ###########################################################################

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    ###########################################################################

    def distance2(self, waypoints, idx1, idx2):

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        dist = dl(waypoints[idx1].pose.pose.position, waypoints[idx2].pose.pose.position)

        return dist
    ###########################################################################
        
    def dist(self, p1, p2):
        """
        dist()
        calculates the euclidian distance between two points
        Input: 
            p1 - first point
            p2 - second point
        Return:
        		dist - distance between the two input points
        """
        return math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2)
    ###########################################################################

    def KMpH_to_MpS(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)
    ###########################################################################


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
