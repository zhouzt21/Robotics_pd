#!/usr/bin/env python3
"""
Launch slam, localization and pose

pgrep -f nav2 | xargs kill
pgrep -f slam | xargs kill

allow us:
- in SLAM mode: update the map, and save it on the disk
- run locailization mode to obtain the pose
"""

import time
import numpy as np
from robotics.ros import ROSNode
from robotics.ros.nav import slam, localization, nav_goal
from robotics.ros.utils import TransformListener
from typing import Tuple, Sequence, Optional, Callable, cast, Dict
from numpy.typing import DTypeLike
from nav2_msgs.srv import LoadMap
from robotics.ros.nav.occupancy_manager import OccupancyGridManager
import rclpy




class Map2D:
    def __init__(self, bbox, resolution, dimension: Tuple[int, ...]=(), dtype: DTypeLike=np.float32) -> None:
        self.minx, self.miny = bbox[0]
        self.maxx, self.maxy = bbox[1]
        self.resolution = resolution

        height = int((self.maxy - self.miny) / resolution)
        width = int((self.maxx - self.minx) / resolution)
        self.map = np.zeros((height, width, *dimension), dtype=dtype)

    def coord_transform(self, x, y):
        return int((x - self.minx) / self.resolution), int((y - self.miny) / self.resolution)

    def inside(self, x, y):
        x, y = self.coord_transform(x, y)
        return x >= self.minx and x < self.maxx and y >= self.miny and y < self.maxy

    def __getitem__(self, key):
        x, y = self.coord_transform(key[0], key[1])
        return self.map[y, x]

    def __setitem__(self, key, value):
        x, y = self.coord_transform(key[0], key[1])
        self.map[y, x] = value


from robotics.sim.ros_plugins import LidarMessage, LaserScan
from robotics.ros.rtypes.msgs import Twist
from nav_msgs.msg import OccupancyGrid

from robotics.ros import rtypes
class MapMsg(rtypes.ROSMsg):
    dtype = OccupancyGrid


class Explorer:
    def __init__(self, bbox, resolution=0.2) -> None:
        node = ROSNode("explorer")
        self.slam_server = slam.SLAMToolbox(node, verbose=False)
        self.navigator = nav_goal.GoalNavigator(node, verbose=False)
        self.listener = TransformListener(node)

        self.resolution = resolution
        self.node = node
        self.visitation = Map2D(bbox, resolution, dtype=np.int64)

        self.lidar = node.create_subscriber('/scan', LidarMessage, queue_size=1)
        self.cmd_vel = node.create_publisher('/cmd_vel', Twist, queue_size=1)

        self.map_request = LoadMap.Request()
        self.map_client = node.create_subscriber('/map', MapMsg, queue_size=1)
        self.map = None

        self.manager = OccupancyGridManager()


    def get_map(self):
        return self.map_client.get()

    def get_loc(self):
        pose = None
        def wait_for_pose():
            nonlocal pose
            out = self.listener('map', 'base_link')
            if out is not None:
                pose = out
                return True
            return False
        self.loop(wait_for_pose)
        assert pose is not None
        import transforms3d
        rpy = transforms3d.euler.quat2euler(pose.q)
        return pose.p[0], pose.p[1], rpy[2]

    def coord_transform(self, x, y):
        pass

    def loop(self, callback, interval=0.05):
        while True:
            now = time.time()
            stop = callback()
            if stop:
                break
            end = time.time()
            if end - now < interval:
                time.sleep(max(0, interval - (end - now)))
            else:
                print('Warning: callback takes too long', end - now)

    def go_forward(self):
        assert self.map is not None
        #self.manager._occ_grid_cb(self.map)
        import queue
        n = 20
        experience = queue.Queue(maxsize=n)
        def check():
            loc = self.get_loc()
            experience.put(loc)

            if experience.full():
                head = experience.get()
                if np.linalg.norm(np.array(head) - np.array(loc)) < 0.1:
                    return True

            self.cmd_vel.publish([0.5, 0., 0.])
            return False
        self.loop(check)


    def turn_left(self, angle: float=np.pi/2):
        init = self.get_loc()
        # loc = list(self.get_loc())
        # loc[2] += angle
        # return self.move_to(*loc)
        def check():
            loc = self.get_loc()
            angle_diff = loc[2] - init[2] - np.pi/2
            if np.abs(angle_diff % (np.pi*2)) < 0.1:
                return True
            self.cmd_vel.publish([0., 0., 1.])

            return False

        self.loop(check)

    def move_to(self, x: float, y: float, theta: float):
        self.navigator.move_to(x, y, theta)

    def main_loop(self):
        start = self.get_loc()
        print('start', start)

        while True:
            self.map = self.get_map()
            print('forward')
            self.go_forward()
            print('turn left')
            self.turn_left()

        self.node.join()



def main():
    explorer = Explorer(((-10, -10), (10, 10)))
    explorer.main_loop()


if __name__ == '__main__':
    main()