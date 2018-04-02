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

LOOKAHEAD_WPS = 200 	# Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # subscriber nodes
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)

        # publisher nodes
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # member variables
        self.current_pose = None
        self.current_velocity = None
        self.current_heading = None
        self.base_waypoints = None
        self.traffic_waypoint = None
        self.traffic_lights = None

        # run process to update waypoints
        self.loop()

    def loop(self):
        """ 
        loop() - continuous loop that processes messages and publishes final waypoints
        """
        rate = rospy.Rate(10)        # 10 Hz  / previously was 2 Hz
        while not rospy.is_shutdown():
            if (self.current_pose is not None) and (self.base_waypoints is not None):
                # get index of next waypoint
                next_index = self.get_next_waypoint(self.current_pose)
                # generate final_waypoints
                final_waypoints = self.get_final_waypoints(next_index)
                # publish message
                self.publish_waypoints(final_waypoints)
                # pause for delay time
                rate.sleep()

    def get_closest_waypoint(self, pose):
        """ 
        get_closest_waypoint(pose)
        gets closest waypoint index, leveraged from path planning
        Input: 
            current pose
        Return: 
            index of closest waypoint
        """
        closest_wp = 1e4        # Any big number

        # sets up pose and waypoints
        p1 = pose.position
        if self.base_waypoints is not None:
            waypoints = self.base_waypoints.waypoints
        else:
            return -1

        # compares each waypoint to current pose
        for i in range(len(waypoints)):
            wp2 = waypoints[i].pose.pose.position
            d = self.dist(p1, wp2)
            if d < closest_wp:
                closest_wp = d
                closest_index = i

        # Check if the closest point is ahead or behind the Ego Car
        closest_coord_vector = np.array([waypoints[closest_index].pose.pose.position.x ,
                                  waypoints[closest_index].pose.pose.position.y])
        prev_coord_vector    = np.array([waypoints[closest_index-2].pose.pose.position.x ,
                                  waypoints[closest_index-2].pose.pose.position.y])

        current_pos_vector   = np.array([p1.x, p1.y])

        DOT_Product_val = np.dot( (closest_coord_vector - prev_coord_vector),
                      (current_pos_vector - closest_coord_vector) )

        if DOT_Product_val > 0:      # The Closest Point is behind the Ego Car
            closest_index = (closest_index + 5) % len(waypoints)   # increment the index by 5 step (by trial and error)

        return closest_index

    def get_next_waypoint(self, pose):
        """ 
        get_next_waypoint(pose)
        gets next waypoint index, leveraged from path planning
        Input: 
            current pose
        Return: 
            index of next waypoint
        """
        # gets closest waypoint
        next_index = self.get_closest_waypoint(pose)
        p1 = pose.position
        p2 = self.base_waypoints.waypoints[next_index].pose.pose.position

        # checks angle and increases index if angle too large
        heading = math.atan2((p2.y - p1.y), (p2.x - p1.x))
        if self.current_heading is not None:
            angle = abs(self.current_heading - heading)
            if angle > math.pi/4:
                next_index += 1

        return next_index

    def get_final_waypoints(self, index):
        """ 
        get_final_waypoints(index)
        gets the waypoints from the index to the number of look ahead waypoints
        Input: 
            index in waypoint list
        Output: 
            list of waypoints to publish
        """
        # setup list
        final_waypoints = []
        wp_len = len(self.base_waypoints.waypoints)  # length of waypoints, and wrap if needed

        # fill waypoint messages and build list
        for i in range(LOOKAHEAD_WPS):
            wp = Waypoint()
            # setup pose 
            wp.pose.pose.position.x = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.position.x
            wp.pose.pose.position.y = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.position.y
            wp.pose.pose.position.z = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.position.z
            # setup orientation
            wp.pose.pose.orientation.x = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.orientation.x
            wp.pose.pose.orientation.y = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.orientation.y
            wp.pose.pose.orientation.z = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.orientation.z
            wp.pose.pose.orientation.w = self.base_waypoints.waypoints[(index+i) % wp_len].pose.pose.orientation.w
            # setup twist 
            wp.twist.twist.linear.x = self.base_waypoints.waypoints[(index+i) % wp_len].twist.twist.linear.x
            wp.twist.twist.linear.y = 0.0
            wp.twist.twist.linear.z = 0.0
            wp.twist.twist.angular.x = 0.0
            wp.twist.twist.angular.y = 0.0
            wp.twist.twist.angular.z = 0.0
            # append to list
            final_waypoints.append(wp)

        return final_waypoints

    def publish_waypoints(self, waypoints):
        """
        publish_waypoints(waypoints)
        sets up the header and publishes the waypoint message
        this is used by the simulator to drive forward at the specified speed.
        Input: 
            waypoints - final set of waypoints
        """
        lane = Lane()
        # setup header and transform
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        # setup list of waypoints
        lane.waypoints = waypoints
        # publish message
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        """
        callback function that sets the current pose of car (position/orientation)
        Input: 
            ROS PoseStamped msg
        """
        self.current_pose = msg.pose
        # calculate the quaternion to get heading
        quaternion = (self.current_pose.orientation.x,
                      self.current_pose.orientation.y,
                      self.current_pose.orientation.z,
                      self.current_pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.current_heading = yaw

    def waypoints_cb(self, waypoints):
        """
        callback function that sets the base waypoints of the capstone track
        Input: 
            list of waypoints
        """
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
        """
        callback function that sets the waypoints of the capstone track traffic
        Input: 
            list of traffic waypoints
        """
        self.traffic_waypoint = msg.data
            
    def velocity_cb(self, msg):
        """
        callback function for the twist message and sets the current velocity
        Input: 
            twist message
        """
        self.current_velocity = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
        
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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
