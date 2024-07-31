from typing import Optional, Tuple
from robotics.ros.ros_node import ROSNode
from ..ros_server import ROSServer
from robotics.cfg import Config
from rclpy.parameter import Parameter


class Localizer(ROSServer):
    cmd = "ros2 launch ros2_nav localization.launch.py"