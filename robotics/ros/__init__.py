from .ros_node import ROSNode

from geometry_msgs.msg import TransformStamped
from robotics import Pose

def transform2pose(msg: TransformStamped):
    import numpy as np
    p = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
    q = np.array([msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z])
    return Pose(
        p, q
    )