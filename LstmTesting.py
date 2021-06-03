#! /usr/bin/env python

# Import libraries and messaging
import keras
import rospy
import numpy as np
import tensorflow as tf
from numpy import inf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tensorflow.keras.models import load_model

# Load neural network model
model = load_model('trainingTP1-2-9-10-14-15OBS1-2-3.h5')

# Summarize model
model.summary()

# Declare variables
i=0
global NewRanges
vel_x_linear = None
vel_z_angular = None
NewRanges = [''] * 90

# Set desired target point
TP_x = 4.0
TP_y = 2.0

x = 0.0
y = 0.0

# Create function to get and select the laser scan values of the mobile robot
def callback(msg):

	# msg.ranges = [] Input list with 360 distance values
	# NewRanges = [] Output list with 90 distance values in the front of the robot (0 to 180 degrees)
	# Change of orientation and values selected every 2 degrees
	for i in range(90):
		if i%2 == 0:
     			NewRanges[i/2] = msg.ranges[i+270]
	for i in range(90):
		if i%2 == 0:
     			NewRanges[(i+90)/2] = msg.ranges[i]
	#To know how many values the list has
	#print len(NewRanges)

# Create function to get the position x,y of the of the mobile robot (tb3_0)
def newOdom(msg):

	global x
	global y
	x = msg.pose.pose.position.x
	y = msg.pose.pose.position.y

# Initialize node
rospy.init_node('testing')

# Subscribe to topics
sub1 = rospy.Subscriber('/tb3_0/scan', LaserScan, callback)
sub2 = rospy.Subscriber('/tb3_0/odom', Odometry, newOdom)

speed = Twist()

# Publish to topics
pub = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size = 1)

# Create while loop
while not rospy.is_shutdown():

	# Get laser values every 0.1 seconds
	ros_rate = rospy.Rate(10) # 10Hz
	ros_rate.sleep()
	if NewRanges[0] != '':
		# Change the infinite values to max sensor distance range (3.5m)
		NN = np.array(NewRanges)
		NN[np.isinf(NN)] = 3.5
		# print(NN)
		# print(NN.shape)
		TPX = np.array(TP_x)
		TPY = np.array(TP_y)
		# print(TPX)
		# print(TPX.shape)
		# print(TPY)
		# print(TPY.shape)
		LTPX = np.append(NN,TPX)
		#print(LTPX)
		LTPXTPY = np.append(LTPX,TPY)
		# print(LTPXTPY)
		# print(LTPXTPY.shape)
		IPX = np.array(x)
		IPY = np.array(y)
		# print(IPX)
		# print(IPX.shape)
		# print(IPY)
		# print(IPY.shape)
		LTPXTPYIPX = np.append(LTPXTPY,IPX)
		# print(LTPXTPYIPX)
		# print(LTPXTPYIPX.shape)
		LTPXTPYIPXIPY = np.append(LTPXTPYIPX,IPY)
		# print(LTPXTPYIPXIPY)
		# print(LTPXTPYIPXIPY.shape)
		New = LTPXTPYIPXIPY.reshape(1, 1, LTPXTPYIPXIPY.shape[0])
		# print(New)
		# print(New.shape)
		print("Target Point:", TP_x, TP_y)
		print("Current Point:", x, y)
		ypredreal = model.predict(New)
		print("Predicted linear and angular velocity:", ypredreal)
		# print(ypredreal.shape)
		# Publish predicted linear and angular speed on Twist
		speed.linear.x = ypredreal[0,0]
		speed.angular.z = ypredreal[0,1]
		pub.publish(speed)
