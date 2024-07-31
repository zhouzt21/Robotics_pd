from typing import Optional, Tuple
from robotics.ros.ros_node import ROSNode
from ..ros_server import ROSServer
from robotics.cfg import Config
from rclpy.parameter import Parameter


class SLAMToolboxConfig(Config):
    odom_frame: str = "odom" 
    map_frame: str = "map"
    base_frame: str = "base_link"
    scan_topic: str = "/scan"
    map_file_name: Optional[str] = None
    map_start_pose: Optional[Tuple[float, float, float]] = None

#from slam_toolbox_msgs.srv import SaveMap as SaveMapSRV
from slam_toolbox.srv import SaveMap, SerializePoseGraph, DeserializePoseGraph, Pause
from geometry_msgs.msg import Pose2D


class SLAMToolbox(ROSServer):
    cmd = 'ros2 launch ros2_nav slam.launch.py'
    config: Optional[SLAMToolboxConfig] = None
    node_name = "/slam_toolbox"

    def __init__(self, node: ROSNode, cmd: str | None=None, verbose: bool=True, config: Optional[SLAMToolboxConfig]=None) -> None:
        super().__init__(node, cmd, verbose)
        self.config = config

        
    def create(self):
        self.map_serializer = self.node.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
        self.map_deserializer = self.node.create_client(DeserializePoseGraph, '/slam_toolbox/deserialize_map')
        self._pauser = self.node.create_client(Pause, '/slam_toolbox/pause_new_measurements')

    def serialize_map(self, path: str):
        # /slam_toolbox/serialize_map
        request = SerializePoseGraph.Request()
        request.filename = path
        return self.map_serializer.call(request)

    def deserialize_map(self, path: str, match_type: int=1, initial_pose: Tuple[float, float, float]=(0., 0., 0.)):
        # /slam_toolbox/deserialize_map	
        request = DeserializePoseGraph.Request()
        request.filename = path
        request.match_type = match_type
        pose = Pose2D()
        pose.x, pose.y, pose.theta = map(float, initial_pose)
        request.initial_pose = pose
        return self.map_deserializer.call(request)