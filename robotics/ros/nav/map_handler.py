from typing import Any
from ..ros_node import ROSNode
import numpy as np
import time
from nav_msgs.msg import OccupancyGrid, MapMetaData
# from map_msgs.msg import OccupancyGridUpdate

class OccupancyMapSubcriber:
    wait_time = 0.1
    _map_msg: OccupancyGrid
    _map_meta_data: MapMetaData
    _map_data: np.ndarray

    def __init__(self, topic, node: ROSNode) -> None:
        self.subscription = node.create_subscription(OccupancyGrid, topic, self._callback, 1)

    def _callback(self, msg: OccupancyGrid):
        self._map_meta_data = msg.info
        self._map_resolution = msg.info.resolution
        self._map_data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._frame_id = msg.header.frame_id
        assert self._frame_id == "map"
        self._map_msg = msg

    def wait_for_map(self):
        while not hasattr(self, "_map_msg"):
            time.sleep(self.wait_time)

    @property
    def map_meta_data(self):
        self.wait_for_map()
        return self._map_meta_data

    @property
    def map_data(self):
        self.wait_for_map()
        return self._map_data


    def get_from_map_x_y(self, x, y):
        map_data = self.map_data
        if self.is_in_gridmap(x, y):
            # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
            # first index is the row, second index the column
            return map_data[y][x]
        else:
            raise IndexError(
                "Coordinates out of gridmap, x: {}, y: {} must be in between: [0, {}], [0, {}]".format(
                    x, y, self.map_meta_data.height, self.map_meta_data.width))

    def __call__(self, x, y) -> Any:
        map_x, map_y = self.get_map_x_y(x, y)
        return self.get_from_map_x_y(map_x, map_y)

    def get_world_x_y(self, costmap_x, costmap_y):
        map_meta_data = self.map_meta_data
        resolution = map_meta_data.resolution
        origin = map_meta_data.origin
        world_x = costmap_x * resolution + origin.position.x
        world_y = costmap_y * resolution + origin.position.y
        return world_x, world_y

    def is_in_gridmap(self, x, y):
        map_meta_data = self.map_meta_data
        if -1 < x < map_meta_data.width and -1 < y < map_meta_data.height:
            return True
        else:
            return False

    def get_map_x_y(self, world_x, world_y):
        map_meta_data = self.map_meta_data
        costmap_x = int(
            np.round((world_x - map_meta_data.origin.position.x) / map_meta_data.resolution))
        costmap_y = int(
            np.round((world_y - map_meta_data.origin.position.y) / map_meta_data.resolution))
        return costmap_x, costmap_y

    @property
    def bbox(self):
        return self.get_world_x_y(0, 0), self.get_world_x_y(self.map_meta_data.width-1, self.map_meta_data.height-1)

        
if __name__ == "__main__":
    import rclpy
    from robotics.ros import ROSNode
    node = ROSNode("test")
    map_sub = OccupancyMapSubcriber("/global_costmap/costmap", node)
    print(map_sub.bbox)

    
    import numpy as np
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-3, 3, 100)
    mm = np.meshgrid(x, y)
    ss = mm[0].shape
    print(mm[0].shape)

    val = [map_sub(x, y) for x, y in zip(mm[0].reshape(-1), mm[1].reshape(-1))]
    val = np.array(val)
    val = val.reshape(*ss)

    import matplotlib.pyplot as plt
    plt.imshow(val)
    plt.show()