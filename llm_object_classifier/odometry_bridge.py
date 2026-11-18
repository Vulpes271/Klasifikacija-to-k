"""Dependency-free planar odometry composition shared with the ROS replay."""
import math


def propagate(anchor, odom_anchor, odom_now):
    dx=odom_now[0]-odom_anchor[0];dy=odom_now[1]-odom_anchor[1]
    yaw=anchor[2]-odom_anchor[2];c=math.cos(yaw);s=math.sin(yaw)
    angle=(anchor[2]+odom_now[2]-odom_anchor[2]+math.pi)%(2*math.pi)-math.pi
    return [float(anchor[0]+c*dx-s*dy),float(anchor[1]+s*dx+c*dy),float(angle)]
