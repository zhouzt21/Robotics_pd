from rclpy.node import Node
from typing import TypedDict
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import os

FILEPATH = os.path.dirname(__file__)

class RGBDepth(TypedDict):
    rgb: np.ndarray
    depth: np.ndarray


def load_board():
    import cv2
    import yaml
    path = os.path.join(FILEPATH, 'charuco.yaml')
    config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    square_length = config['squareLength']
    board_length = config['markerLength']
    board = cv2.aruco.CharucoBoard((5, 7), square_length, board_length , ARUCO_DICT)
    return board


def decode(msg: Image):
    encoding = msg.encoding
    #rgb = CvBridge().imgmsg_to_cv2(msg[0], desired_encoding='passthrough')

    dim = int(encoding[-1])
    if encoding.startswith('8U'):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, dim))[..., :3]
    elif encoding.startswith('32F'):
        img = (np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width, dim)) * 255).astype(np.uint8)[..., :3]
    elif encoding.startswith('16U'):
        assert dim == 1
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
    else:
        raise NotImplementedError(f'encoding {encoding} not supported')
    return img


class RGBDepthSubscriber:
    def __init__(self, node: Node, callback=None) -> None:
        self.node = node

        self.rgb_sub = message_filters.Subscriber(node, Image, '/rgb')
        self.depth_sub = message_filters.Subscriber(node, Image, '/depth')
        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        if callback is not None:
            self.synchronizer.registerCallback(callback)
    

    def extract_rgb_depth(self, msg_) -> RGBDepth:
        return {
            'rgb': decode(msg_[0]),
            'depth': decode(msg_[1])
        }


    def register_callback(self, callback):
        self.synchronizer.registerCallback(callback)



def rgb_depth2pointcloud(rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, max_depth=None, obj_mask=None, return_o3d=False):
    mask = depth > 0
    if obj_mask is not None:
        mask = mask & obj_mask

    if max_depth is not None:
        mask = mask & (depth < max_depth * 1000)

    width = rgb.shape[1]
    height = rgb.shape[0]

    coords = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(np.float32)
    coords = coords[:,::-1] + 0.5

    coords = np.concatenate((coords, np.ones((height, width, 1))), axis=-1)
    coords = coords @ np.linalg.inv(intrinsic).T


    position = coords * depth[..., None]
    depth = - position[mask]/1000.
    rgb = rgb[mask]

    xyz = depth.reshape(-1, 3).copy()
    if return_o3d:
        assert rgb.dtype == np.uint8, rgb.dtype
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb/255.)
        return pcd

    xyz = xyz[:, [2, 0, 1]]
    xyz[:, 0] *= -1
    xyz[:, 1] *= -1

    return xyz, rgb.reshape(-1, 3)

    
from sensor_msgs.msg import CameraInfo

class CameraInfoPublisher:
    def __init__(self, node: Node, K, frame: str) -> None:
        self.node = node
        camera_info = CameraInfo()
        camera_info.k = K.flatten().tolist()
        camera_info.header.frame_id = frame
        self.camera_info = camera_info

        from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = node.create_publisher(CameraInfo, '/camera_info', qos)
        self.publish_info()

    def publish_info(self):
        self.publisher.publish(self.camera_info)

        
        
def pub_once(msg, topic, name='pub', dt=0.01, counter=1):
    import rclpy

    rclpy.init()
    node = rclpy.create_node(name) 
    pub = node.create_publisher(msg.__class__, topic, 10) # "joint_action"
    counter = counter
    def timer_callback():
        nonlocal counter
        print('Publishing: "{}" to {}'.format(msg, topic))
        pub.publish(msg)
        counter = counter - 1
        #exit = True
    timer = node.create_timer(dt, timer_callback)
    while rclpy.ok() and counter > 0:
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()