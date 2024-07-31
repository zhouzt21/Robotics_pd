from typing import Any
import numpy as np

from tf2_msgs.msg import TFMessage as ROSTFMessage
from geometry_msgs.msg import TransformStamped
from .format import Format

class Transform(Format):
    dtype = TransformStamped

    @classmethod 
    def from_msg(cls, msg: TransformStamped):
        p = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        q = np.array([msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z])
        parent_frame = msg.header.frame_id
        child_frame = msg.child_frame_id
        return {'p': p, 'q': q, 'parent': parent_frame, 'child': child_frame}

    @classmethod
    def to_msg(cls, t):
        p, q, parent_frame, child_frame = t['p'], t['q'], t['parent'], t['child']

        msg = TransformStamped()
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, p)
        msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z = map(float, q)
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        return msg

        

class TFMessage(Format):
    dtype = ROSTFMessage

    @classmethod
    def from_msg(cls, msg: Any):
        return [Transform.from_msg(i) for i in msg.transforms]

    @classmethod
    def to_msg(cls, t):
        msg = ROSTFMessage()
        msg.transforms = [Transform.to_msg(i) for i in t]
        return msg

    @classmethod
    def add_stamp(cls, msg: ROSTFMessage, stamp):
        for i in msg.transforms:
            Transform.add_stamp(i, stamp)
        return msg

        

from geometry_msgs.msg import Twist as ROSTwist

class Twist(Format):
    dtype = ROSTwist

    @classmethod
    def from_msg(cls, msg: ROSTwist):
        return np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    @classmethod
    def to_msg(cls, t):
        msg = ROSTwist()
        msg.linear.x, msg.linear.y, msg.angular.z = t
        return msg

    @classmethod
    def add_stamp(cls, msg: ROSTwist, stamp):
        return msg