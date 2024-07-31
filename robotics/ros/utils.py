"""_summary_
some simple tools that I am not sure where I put it. 
"""

import time
from rclpy.time import Time
from .ros_node import ROSNode
from robotics import Pose
from tf2_ros import Buffer, TransformListener as ROS_TransformListener
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import TransformStamped
from rclpy.time import Duration
from datetime import timedelta
from typing import Optional



class TransformListener:
    def __init__(self, node: ROSNode, window: float=0.05) -> None:
        self.tf_buffer = Buffer()
        self.node = node
        self.window = window
        self.listener = ROS_TransformListener(self.tf_buffer, node)

    def __call__(self, parent: str, child: str, window: Optional[float]=None):
        raise NotImplementedError("This function is not implemented yet!")
        now = self.node.get_time()
        nano = now.nanoseconds
        nano = max(nano - int((window or self.window) * 1e9), 0)
        before = Time(nanoseconds=nano)

        try:
            transform = self.tf_buffer.lookup_transform(parent, child, before) # , Duration(seconds=0.1)
        except:
            return None

        out = Transform.from_msg(transform)
        return Pose(out['p'], out['q'])


    @staticmethod
    def test():
        # has to run some navigation stack
        node = ROSNode('test_transform_lookup')

        lookuper = TransformListener(node)
        while True:
            print(lookuper('map', 'base_link'))
            time.sleep(1)



        
if __name__ == "__main__":
    TransformListener.test()