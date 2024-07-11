"""_summary_
ROS publisher for publishing sapien data to ROS topics
"""
import numpy as np
from rosgraph_msgs.msg import Clock
from sapien import physx
from rclpy.time import Time
import sapien
from typing import List, Sequence, Tuple, Optional, Callable, cast, Dict, Union
from robotics import Pose
#from ..simulator import ROSModule, FrameLike
from .module import ROSModule, FrameLike
from robotics.ros import ROSNode




class ROSPlugin:
    frame_mapping: Dict[str, str]
    topic_mapping: Callable[[str], str]
    def __init__(self, frame_mapping: Optional[Dict[str, str]]=None) -> None:
        self.frame_mapping = frame_mapping or {}
        self.enabled = True

    def _load(self, world: ROSModule, node: ROSNode):
        self.node = node
        self.world = world

    def before_control_step(self, world: ROSModule):
        pass

    def after_step(self, world: ROSModule):
        pass

    def topic(self, name):
        """
        """
        if not hasattr(self, 'topic_mapping'):
            self.topic_mapping = lambda x: x
        return self.topic_mapping(name)



#from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from std_msgs.msg import String

class RobotNamePublisher(ROSPlugin):
    def __init__(self, name: str, frame_mapping: Optional[Dict[str, str]] = None) -> None:
        super().__init__(frame_mapping)
        self.name = name


    def _load(self, world: ROSModule, node: ROSNode):
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.robot_name_publisher = node.create_publisher(
            String, self.topic('/robot_name_static'), qos
        )
        self.robot_name_publisher.publish(String(data=self.name))



class TimePublisher(ROSPlugin):
    def _load(self, world: ROSModule, node: ROSNode):
        node.get_logger().info('Time publisher loaded')
        self.publisher = node.create_publisher(Clock, self.topic('/clock'), 1)
        self.cur = 1234

    def after_step(self, world: ROSModule):
        msg = Clock()
        nanoseconds = self.cur

        seconds = int(nanoseconds / 1000000000)
        nanoseconds = int(nanoseconds % 1000000000)
        msg.clock = Time(nanoseconds=int(seconds * 1e9) + nanoseconds).to_msg()
        self.publisher.publish(msg)
        self.cur += world.sim.dt * 1e9


from robotics.ros.msgs.transform import TFMessage, Transform
class TFPublisher(ROSPlugin):
    """_summary_
    publish tf transformation to ROS topics  
    dynamic and static are lists of tuples, each tuple is a pair of frame names.
    the frame names should be able to be found by the simulator.find()
    frame_mapping is a dictionary that maps the frame name in the simulator to the frame name in ROS
    """
    def __init__(
        self,
        dynamic: Sequence[Tuple[FrameLike, FrameLike]] = (),
        static: Sequence[Tuple[FrameLike, FrameLike]] = (),
        frame_mapping: Optional[Dict[str, str]] = None,
        topic_name: str = '/tf',
    ):
        self.dynamic = dynamic
        self.static = static
        self.topic_name = topic_name
        super().__init__(frame_mapping)

    def get_by_frames(self, frame1: FrameLike, frame2: FrameLike, world: ROSModule):
        frame1, pose1 = world.get_frame(frame1, self.frame_mapping)
        frame2, pose2 = world.get_frame(frame2, self.frame_mapping)
        rel_pose = pose1.inv() * pose2
        return {"p": rel_pose.p, "q": rel_pose.q, "parent": frame1, "child": frame2}

    def _load(self, world: ROSModule, node: ROSNode):
        super()._load(world, node)
        node.get_logger().info('TF publisher loaded')
        self.world = world
        self.publisher = node.create_publisher(TFMessage.dtype, self.topic(self.topic_name), 10)
        tfs = [Transform.to_msg(self.get_by_frames(*i, world)) for i in self.static]
        print(tfs)
        node.static_broadcaster.sendTransform(tfs)

    def before_control_step(self, world: ROSModule):
        pass

    def after_step(self, world: ROSModule):
        tfs = []
        for i in self.dynamic:
            tfs.append(self.get_by_frames(*i, world))
        msg = TFMessage.to_msg(tfs)
        msg = TFMessage.add_stamp(msg, self.node.get_clock().now().to_msg())
        self.publisher.publish(msg)


        
from ..sensors.depth_camera import StereoDepthCamera
class RGBDPublisher(ROSPlugin):
    def __init__(self, depth: StereoDepthCamera, frame_name: Optional[str], frame_mapping: Optional[Dict[str, str]] = None, frequence: int=1) -> None:
        super().__init__(frame_mapping)
        self.depth = depth
        self.frame_name = frame_name
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.frequence = frequence
        self.counter = -1

    def _load(self, world: ROSModule, node: ROSNode):
        from sensor_msgs.msg import Image
        super()._load(world, node)
        node.get_logger().info('RGBD publisher loaded')

        self.publisher = node.create_publisher(Image, self.topic('/rgb'), 10)
        self.depth_publisher = node.create_publisher(Image, self.topic('/depth'), 10)

        from robotics.utils.camera_utils import CameraInfoPublisher
        assert self.frame_name is not None
        self.camera_info = CameraInfoPublisher(node, self.depth.get_camera_params()['intrinsic_cv'], self.frame_name)

    def after_step(self, world: ROSModule):
        self.counter += 1
        if self.counter % self.frequence != 0:
            return
        output = self.depth.get_observation()
        RGB = output['Color'][..., :3]
        depth = output['depth']
        # assert RGB.max() < 1.1

        RGB = (RGB * 255).astype(np.uint8)

        assert RGB.dtype == np.uint8

        rgb = RGB
        msg =  self.bridge.cv2_to_imgmsg(rgb, encoding=f'8UC{rgb.shape[-1]}')
        timestamp = self.node.get_clock().now().to_msg()
        msg.header.stamp = timestamp
        self.publisher.publish(msg)

        depth = ((depth) * 1000).astype(np.uint16)

        msg = self.bridge.cv2_to_imgmsg(depth, encoding=f'16UC1')
        msg.header.stamp = timestamp
        self.depth_publisher.publish(msg)

        

from sensor_msgs.msg import LaserScan, PointCloud2

    
from ..sensors.lidar_sim import Lidar
from ..sensors.lidar_v2 import Lidar as LidarV2
class LidarPublisher(ROSPlugin):
    def __init__(self, lidar: Union[Lidar, LidarV2], frame: FrameLike, frame_mapping: Optional[Dict[str, str]]=None) -> None:
        super().__init__(frame_mapping)
        self.lidar = lidar
        self.frame = frame

    def _load(self, world: ROSModule, node: ROSNode):
        super()._load(world, node)
        node.get_logger().info('Lidar publisher loaded')
        self.publisher = node.create_publisher(LaserScan, self.topic('/scan'), 10)

    def depth2pts(self, obs, camera_params):
        position = obs["Position"]
        position = position[position[..., 2] < 0]
        position[..., 3] = 1
        import torch
        # print(camera_pose, 'pose')
        cam2world = torch.tensor(camera_params["cam2world_gl"]).cuda()
        xyzw = torch.tensor(position).cuda().reshape(-1, 4) @ cam2world.T
        xyzw = xyzw[xyzw[..., 2] < 0.1]
        xyzw = xyzw[xyzw[..., 2] > 0.02]
        return xyzw

    def after_step(self, world: ROSModule):
        import torch
        frame_id, pose = world.get_frame(self.frame, self.frame_mapping)
        pts = []
        if isinstance(self.lidar, Lidar):
            for _, i in self.lidar.cameras.items():
                obs = i.get_observation()
                param = i.get_camera_params()
                pts.append(self.depth2pts(obs, param))
            points = torch.cat(pts, dim=0) # changed in maniskill3
        else:
            points = torch.tensor(self.lidar.get_observation()['points'], dtype=torch.float32).cuda()
        laser_scane_msg = self.pointcloud2laser(points, pose, frame_id)
        self.publisher.publish(laser_scane_msg)


    def pointcloud2laser(self, points_, base_pose: sapien.Pose, frame_id: str):
        import torch

        xyzw: torch.Tensor = points_ 
        from sensor_msgs.msg import LaserScan

        xyzw = xyzw @ torch.tensor(base_pose.inv().to_transformation_matrix()).cuda().T

        #theta 
        dist = (xyzw[:, :3] ** 2).sum(dim=1) ** 0.5
        theta = torch.arctan2(xyzw[:, 1], xyzw[:, 0]) # not sure about the ROS's coordinate system
        bins = torch.linspace(0., np.pi * 2, 360)

        index = ((theta + np.pi * 2) % (np.pi * 2) /  (np.pi * 2) * len(bins)).long().clamp(0, len(bins) - 1)


        try:
            from torch_scatter import scatter_min # type: ignore
        except ImportError:
            raise ImportError('Please install torch_scatter: https://github.com/rusty1s/pytorch_scatter.')
        out = torch.zeros(len(bins)).cuda()
        out.fill_(np.inf)
        out, _ = scatter_min(dist, index, dim=0, dim_size=len(bins), out=out)

        laser_data = LaserScan()
        laser_data.header.frame_id = frame_id  # Set the frame ID

        laser_data.range_min = 0.2  # Minimum range value
        laser_data.range_max = 10.0  # Maximum range value

        laser_data.angle_min = 0.
        laser_data.angle_max = np.pi * 2
        laser_data.angle_increment = np.pi * 2 / (len(bins) - 1)
        laser_data.ranges = out.detach().cpu().numpy().tolist()
        laser_data.intensities = [30000. if laser_data.ranges[i] < 100. else 0.  for i in range(len(bins))]

        laser_data.header.stamp = self.node.get_clock().now().to_msg()

        return laser_data

        
from geometry_msgs.msg import Twist
class CMDVel(ROSPlugin):
    """_summary_
    Taking cmd val as input and directly override the control signal for the mobile base.
    We consider it as the interface to control the hardware so it has high priority and will override other control signals.
    """
    def __init__(self, action_decay: float = 1.) -> None:
        super().__init__()
        # HACK: in case that the action is not set, we set it to zero manually
        #   needs better solution and should be removed in the future
        self.action_decay = action_decay
    
    def _load(self, world: ROSModule, node: ROSNode):
        super()._load(world, node)
        node.get_logger().info('CMDVel publisher loaded')
        self.twist = None
        def update_vel(msg: Twist):
            self.twist = msg
        self.subscriber = node.create_subscription(Twist, '/cmd_vel', update_vel, 10)
        self.robot = world.robot
        assert self.robot is not None, 'robot is not loaded'
        self.base_controller = self.robot.controllers['base']
        self.last_action = np.zeros(3)

    def before_control_step(self, obs, **kwargs):
        if self.twist is not None:
            twist: Twist = self.twist
            x, y, rot = twist.linear.x, twist.linear.y, twist.angular.z
            action = np.array([x, y, rot])
            self.last_action = self.base_controller.inv_scale_action(action)
            self.base_controller.set_action(self.last_action)

            self.twist = None
        else:
            self.last_action *= self.action_decay
            self.base_controller.set_action(self.last_action)

            
            
from std_msgs.msg import Float64MultiArray, Float64
class ArmServo(ROSPlugin):
    """_summary_
    Taking cmd val as input and directly override the control signal for the mobile base.
    We consider it as the interface to control the hardware so it has high priority and will override other control signals.
    """
    def __init__(self, controller) -> None:
        super().__init__()
        from ..robot.posvel_controller import PDJointPosVelController
        self.controller: PDJointPosVelController = controller

        self.gripper_controller = self.controller.robot.controllers['gripper']

    
    def _load(self, world: ROSModule, node: ROSNode):
        super()._load(world, node)
        node.get_logger().info('Arm servo publisher loaded')

        self.publisher = node.create_publisher(Float64MultiArray, self.topic("/joint_states"), 10)
        self.dim = self.controller.action_space.shape[0] - 1
        def send_radiance(msg):
            assert len(msg.data) == self.dim + 1
            radiance = msg.data[:self.dim]
            speed = int(msg.data[-1])
            if speed <= 0 or speed > 100:
                print('invalid speed', speed)
            else:
                action = self.controller.inv_scale_action(np.append(radiance, speed))
                self.controller.set_action(action)

        self.subscriber = node.create_subscription(Float64MultiArray, self.topic("/joint_action"), send_radiance, 10)
        robot = world.robot
        assert robot is not None, 'robot is not loaded'
        self.robot = robot
        self.arm_controller = self.robot.controllers['arm']

        def send_gripper(msg):
            self.gripper_controller.set_action(np.array([msg.data]))

        self.gripper_subscriber = node.create_subscription(Float64, self.topic("/gripper_action"), send_gripper, 10)
        self.gripper_publisher = node.create_publisher(Float64MultiArray, self.topic("/gripper_state"), 10)


    def after_step(self, obs, **kwargs):
        msg = Float64MultiArray()
        msg.data = [float(i) for i in self.arm_controller.qpos]
        self.publisher.publish(msg)
        msg = Float64MultiArray()
        msg.data = [float(i) for i in self.robot.articulation.get_qpos()[-6:]] #TODO: fix this
        self.gripper_publisher.publish(msg)
