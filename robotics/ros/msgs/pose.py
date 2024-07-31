from typing import Any
import numpy as np
from geometry_msgs.msg import Pose2D as ROSPose2D, PoseStamped as ROSPoseStamped
from .format import Format


class Pose2D(Format):
    dtype = ROSPose2D
    
    @classmethod
    def from_msg(cls, msg: ROSPose2D):
        return msg.x, msg.y, msg.theta
        
    @classmethod
    def to_msg(cls, t):
        msg = ROSPose2D()
        msg.x, msg.y, msg.theta = t
        return msg

        
    
    
class PoseStamped(Format):
    dtype = ROSPoseStamped

    
    @classmethod
    def from_msg(cls, msg: ROSPoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        frame = msg.header.frame_id
        return {'p': p, 'q': q, 'frame': frame}
        
    @classmethod
    def to_msg(cls, t):
        p, q, frame = t['p'], t['q'], t['frame']

        msg = ROSPoseStamped()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = map(float, q)
        msg.header.frame_id = frame

        return msg