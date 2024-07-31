#import numpy as np
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from typing import Any, Optional
import transforms3d


def stamp2datetime(data):
    timestamp = datetime.datetime.utcfromtimestamp(data.header.stamp.to_sec())
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

def plot_laser_scane(ax: plt.Axes, data: LaserScan, world):
    odom = world.get('odom', None)
    if odom is None:
        logging.warning("No odom found, skipping laser scan")
        return


    origin = np.array(odom[:2])
    theta = odom[2]
    angles = [data.angle_min + i * data.angle_increment for i in range(len(data.ranges))]
    x = [origin[0] + data.ranges[i] * np.cos(angles[i] + theta) for i in range(len(data.ranges)) if data.intensities[i] > 0.]
    y = [origin[1] + data.ranges[i] * np.sin(angles[i] + theta) for i in range(len(data.ranges)) if data.intensities[i] > 0.]
    ax.scatter(x, y, s=1)  # s is the marker size

    
def plot_xytheta(ax: plt.Axes, data: TransformStamped, world):
    print(stamp2datetime(data))
    x, y = data.transform.translation.x, data.transform.translation.y
    quat = np.array([data.transform.rotation.w, data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z])

    # quat to get rotation about z
    #theta = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
    theta = transforms3d.euler.quat2euler(quat)[2]

    if data.header.frame_id == 'odom':
        world['odom'] = [x, y, theta]
        # Create a scatter plot of the laser scan data
        ax.scatter(x, y, s=1)  # s is the marker size
        ax.arrow(x, y, np.cos(theta), np.sin(theta), width=0.01)


def plot_tfmessage(ax: plt.Axes, data: TFMessage, world):
    if data.transforms:
        for transform in data.transforms:
            plot_xytheta(ax, transform, world)


def plot_msg(ax: plt.Axes, data: Any, world, topic: Optional[str]=None):
    if isinstance(data, LaserScan) or topic == '/scan':
        plot_laser_scane(ax, data, world)
    elif isinstance(data, TransformStamped):
        plot_xytheta(ax, data, world)
    elif isinstance(data, TFMessage) or topic == '/tf':
        plot_tfmessage(ax, data, world)
    else:
        raise NotImplementedError(f"Cannot plot data of type {type(data)}")