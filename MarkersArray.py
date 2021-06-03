#! /usr/bin/env python

# Import libraries and messaging
import math
import rospy
import roslib; roslib.load_manifest('visualization_marker_tutorials')
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Create function to get the position x,y of the of the mobile robot (tb3_0)
def newOdom(msg):

	global x
	global y
	x = msg.pose.pose.position.x
	y = msg.pose.pose.position.y

# Subscribe to topics
sub = rospy.Subscriber('/tb3_0/odom', Odometry, newOdom)

# Create and publish new topic
topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, MarkerArray, queue_size=10)

# Initialize node
rospy.init_node('markers')

# Declare variables
markerArray = MarkerArray()
count = 0
MARKERS_MAX = 100

# Create while loop
while not rospy.is_shutdown():

   marker = Marker()
   marker.header.frame_id = "tb3_0/odom"
   marker.type = marker.SPHERE
   marker.action = marker.ADD
   marker.scale.x = 0.1
   marker.scale.y = 0.1
   marker.scale.z = 0.1
   marker.color.a = 1.0
   marker.color.r = 1.0
   marker.color.g = 1.0
   marker.color.b = 0.0
   # marker.lifetime = 100.0
   marker.pose.orientation.w = 1.0
   marker.pose.position.x = x
   marker.pose.position.y = y
   # marker.pose.position.z = math.cos(count / 30.0)

   # We add the new marker to the MarkerArray, removing the oldest
   # marker from it when necessary

   if(count > MARKERS_MAX):
       markerArray.markers.pop(0)

   markerArray.markers.append(marker)

   # Renumber the marker IDs
   id = 0
   for m in markerArray.markers:
       m.id = id
       id += 1

   # Publish the MarkerArray
   publisher.publish(markerArray)

   count += 1
   rospy.sleep(0.01)
