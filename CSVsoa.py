#! /usr/bin/env python

# Import libraries
import io
import csv
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from nav_msgs.msg import Odometry

# Write in .csv
csv_data = io.BytesIO()
csv_writer = csv.writer(csv_data)

# Create function to get the position x,y of the of the mobile robot (tb3_0)
def newOdom(msg):

	global x
	global y
	x = msg.pose.pose.position.x
	y = msg.pose.pose.position.y

# Initialize node
rospy.init_node('store_values')

# Subscribe to topics
sub = rospy.Subscriber('/tb3_0/odom', Odometry, newOdom)

# Create while loop
while not rospy.is_shutdown():

	# Store values in a .csv file every 0.1 seconds
    ros_rate = rospy.Rate(10) # 10Hz
    ros_rate.sleep()
    execTime=np.empty((0, 2), float)
    execTime = np.append(execTime, np.array([[x,y]]), axis=0)
    print(execTime)
    for i, data in enumerate(execTime):
        with open('ExecTimesA6.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print("All data is saved!")


    # with open('storedvalues.csv', 'a') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(x)
    #     writer.writerow(y)
