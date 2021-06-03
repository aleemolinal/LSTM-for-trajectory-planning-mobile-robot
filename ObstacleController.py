#! /usr/bin/env python

# Import libraries and messaging
import time
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

# Declare variables
x = 0.0
y = 0.0

# Create function to get the position x,y of the dynamic obstacle (tb3_1)
def newOdom (msg):

	global x
	global y
	x = msg.pose.pose.position.x
	y = msg.pose.pose.position.y

# Initialize node
rospy.init_node('speed_controller')

# Subscribe and publish to topics
sub = rospy.Subscriber('/tb3_1/odom', Odometry, newOdom)
pub = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size = 1)

speed = Twist()
r = rospy.Rate(1) # 1 Hz

# Set desired goal point
goal = Point ()
goal.x = 2.5
goal.y = 0.5
#
# Sleep
time.sleep(4.5)

# Create while loop
while not rospy.is_shutdown():

	# Publish desired linear velocity in x for the dynamic obstacle (tb3_1)
	inc_x = abs(goal.x - x)
	inc_y = abs(goal.y - y)
	if inc_x < 0.2 and inc_y < 0.2:
		speed.linear.x = 0.0
	else:
		speed.linear.x = 0.2
	# print(x)
	# print(y)
	pub.publish(speed)
	r.sleep()
