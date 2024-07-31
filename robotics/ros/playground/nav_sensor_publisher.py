"""
simple code to publish everything to ROS nav stack
- laser scan
- point cloud
- tf
- odom
- map
"""
import torch

import numpy as np
import sapien.core as sapien
from .publisher import Publisher, PublisherConfig, Optional

from robotics.utils import logger
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from . import ros_api


def pose2msg(p, q, header, child, time):
    A = TransformStamped()
    A.header.stamp = time
    A.header.frame_id = 'odom'
    A.child_frame_id = 'base_footprint'
    A.transform.translation.x = float(p[0])
    A.transform.translation.y = float(p[1])
    A.transform.translation.z = float(p[2])
    A.transform.rotation.w = float(q[0])
    A.transform.rotation.x = float(q[1])
    A.transform.rotation.y = float(q[2])
    A.transform.rotation.z = float(q[3])
    return A

def create_tf_msgs(*pose):
    tf = TFMessage()
    for p in pose:
        tf.transforms.append(p)
    return tf


def pose2tf(pose, pose_local, time, add_odom):

    #import rospy
    tf = TFMessage()
    A = TransformStamped()
    A.header.stamp = time
    A.header.frame_id = 'odom'
    A.child_frame_id = 'base_footprint'
    A.transform.translation.x = float(pose.p[0])
    A.transform.translation.y = float(pose.p[1])
    A.transform.translation.z = float(pose.p[2])
    A.transform.rotation.w = float(pose.q[0])
    A.transform.rotation.x = float(pose.q[1])
    A.transform.rotation.y = float(pose.q[2])
    A.transform.rotation.z = float(pose.q[3])
    # if add_odom:
    tf.transforms.append(A)

    C = TransformStamped()
    C.header.stamp = time
    C.header.frame_id = 'base_footprint'
    C.child_frame_id = 'base_link'
    C.transform.translation.x = 0.
    C.transform.translation.y = 0.
    C.transform.translation.z = 0.
    C.transform.rotation.w = 1.
    C.transform.rotation.x = 0.
    C.transform.rotation.y = 0.
    C.transform.rotation.z = 0.
    tf.transforms.append(C)

    B = TransformStamped()
    B.header.stamp = time
    B.header.frame_id = 'base_link'
    B.child_frame_id = 'base_laser'
    B.transform.translation.x = float(pose_local.p[0])
    B.transform.translation.y = float(pose_local.p[1])
    B.transform.translation.z = float(pose_local.p[2])
    B.transform.rotation.w = float(pose_local.q[0])
    B.transform.rotation.x = float(pose_local.q[1])
    B.transform.rotation.y = float(pose_local.q[2])
    B.transform.rotation.z = float(pose_local.q[3])
    tf.transforms.append(B)

    D = TransformStamped()
    D.header.stamp = time
    D.header.frame_id = 'base_link'
    D.child_frame_id = 'base_sensor'
    D.transform.translation.x = float(pose_local.p[0])
    D.transform.translation.y = float(pose_local.p[1])
    D.transform.translation.z = float(pose_local.p[2])
    D.transform.rotation.x = float(pose_local.q[1])
    D.transform.rotation.y = float(pose_local.q[2])
    D.transform.rotation.z = float(pose_local.q[3])
    D.transform.rotation.w = float(pose_local.q[0])
    tf.transforms.append(D)


    return tf


def build_point_cloud_msg(points, parent_frame):
    # https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2
    import std_msgs.msg as std_msgs

    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    from sensor_msgs.msg import PointField
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )


def depth2pointcloud_msg(obs, camera_params, camera_pose: sapien.Pose):
    position = obs["Position"]
    cam2world = camera_params["cam2world_gl"]
    rgba = obs["Color"]
    mask = (position[..., 2] < 0).reshape(-1)
    position = position.reshape(-1, 4)[mask]
    rgba = rgba.reshape(-1, 4)[mask]
    position[..., 3] = 1
    pointcloud = position.reshape(-1, 4) @ (cam2world.T @ camera_pose.inv().to_transformation_matrix().T)
    pointcloud = np.concatenate([pointcloud[..., :3], rgba], axis=1)
    return build_point_cloud_msg(pointcloud, 'base_sensor')


def depth2pts(obs, camera_params):
    position = obs["Position"]
    position = position[position[..., 2] < 0]
    position[..., 3] = 1
    import torch
    # print(camera_pose, 'pose')
    cam2world = torch.tensor(camera_params["cam2world_gl"]).cuda()
    xyzw = torch.tensor(position).cuda().reshape(-1, 4) @ cam2world.T
    xyzw = xyzw[xyzw[..., 2] < 0.2]
    xyzw = xyzw[xyzw[..., 2] > 0.05]
    return xyzw


def pointcloud2laser(xyzw: torch.Tensor, base_pose: sapien.Pose, verbose=False):
    from sensor_msgs.msg import LaserScan
    # print('camera_base', base_pose)
    xyzw = xyzw @ torch.tensor(base_pose.inv().to_transformation_matrix()).cuda().T

    #theta 
    dist = (xyzw[:, :3] ** 2).sum(axis=1) ** 0.5
    theta = torch.arctan2(xyzw[:, 1], xyzw[:, 0]) # not sure about the ROS's coordinate system
    bins = torch.linspace(0., np.pi * 2, 360)

    index = ((theta + np.pi * 2) % (np.pi * 2) /  (np.pi * 2) * len(bins)).long().clamp(0, len(bins) - 1)


    from torch_scatter import scatter_min
    out = torch.zeros(len(bins)).cuda()
    out.fill_(np.inf)
    out, _ = scatter_min(dist, index, dim=0, dim_size=len(bins), out=out)

    laser_data = LaserScan()
    laser_data.header.frame_id = 'base_laser'  # Set the frame ID

    laser_data.range_min = 0.2  # Minimum range value
    laser_data.range_max = 10.0  # Maximum range value

    laser_data.angle_min = 0.
    laser_data.angle_max = np.pi * 2
    laser_data.angle_increment = np.pi * 2 / (len(bins) - 1)
    laser_data.ranges = out.detach().cpu().numpy().tolist()
    laser_data.intensities = [30000. if laser_data.ranges[i] < 100. else 0.  for i in range(len(bins))]

    return laser_data


class NavSensorPublisherConfig(PublisherConfig):
    name: str = 'depth_publisher'
    channel: str = '/base_scan,/base_cloud,/tf'
    queue_size: Optional[int] = 100000000
    rate: float = 10.
    cmd: Optional[str] = None


def build_odom_msg(pose: sapien.Pose, vel):
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Quaternion, Pose, Twist, Vector3
    import transforms3d

    odom = Odometry()
    odom.header.stamp = ros_api.get_time()
    odom.header.frame_id = "map"
    odom.child_frame_id = "odom"
    pose_msg = Pose()
    pose_msg.position.x = float(pose.p[0])
    pose_msg.position.y = float(pose.p[1])

    pose_msg.orientation = Quaternion(
        x=float(pose.q[1]),
        y=float(pose.q[2]),
        z=float(pose.q[3]),
        w=float(pose.q[0])
    )
    odom.pose.pose = pose_msg

    vel = list(map(float, vel))
    x, y = vel[:2]
    x, y = transforms3d.quaternions.rotate_vector([x, y, 0], pose.inv().q)[:2]

    twist = Twist()
    twist.linear = Vector3(x=x, y=y, z=0.)
    twist.angular = Vector3(x=0., y=0., z=vel[2])
    odom.twist.twist = twist
    return odom

    

class NavSensorPublisher(Publisher):
    def get_data_format(self):
        from sensor_msgs.msg import LaserScan, PointCloud2
        from tf2_msgs.msg import TFMessage
        from nav_msgs.msg import Odometry
        return [TFMessage, Odometry, LaserScan, PointCloud2]

    def fn(self, data):

        (laser, pointcloud), (pose, pose_local), vel, add_odom = data

        cur_time = ros_api.get_time()
        pose_msg = pose2tf(pose, pose_local, cur_time, add_odom=add_odom)

        laser.header.stamp = cur_time
        pointcloud.header.stamp = cur_time
        odom_msg = build_odom_msg(pose, vel)
        odom_msg.header.stamp = cur_time
        return pose_msg, odom_msg, laser, pointcloud