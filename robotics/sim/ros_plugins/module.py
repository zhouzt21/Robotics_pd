import numpy as np
from sapien import Pose, physx, Entity
from typing import Optional, List, cast, Any, TYPE_CHECKING

from ..simulator_base import FrameLike
from ..simulator import Simulator
from ..sensors.sensor_base import SensorBase


from robotics.ros import ROSNode
from robotics.utils.sapien_utils import get_rigid_dynamic_component

if TYPE_CHECKING:
    from .ros_plugin import ROSPlugin


class ROSModule:
    sim: "Simulator"
    def __init__(self, name: str, use_sim_time: bool = True, args: Optional[list[Any]]=None, ros_domain_id: Optional[int] = None) -> None:
        self.ros_node = ROSNode(name, use_sim_time, args, ros_domain_id)
        self.ros_plugins: List["ROSPlugin"] = []

    def set_sim(self, sim: "Simulator"):
        from . import TimePublisher # avoid circular import
        self.sim = sim
        self.robot = sim.robot
        self._elem_cache = {}
        if self.robot is not None:
            self.ros_plugins.append(TimePublisher())
            self.ros_plugins += self.robot.get_ros_plugins()


    def get_active_ros_plugins(self):
        for plugin in self.ros_plugins:
            if plugin.enabled:
                yield plugin

    def before_control_step(self):
        if self.ros_node is not None:
            for plugin in self.get_active_ros_plugins():
                plugin.before_control_step(self)
            
    def after_control_step(self):
        for plugin in self.get_active_ros_plugins():
            plugin.after_step(self)

    
    def load(self):
        for plugin in self.get_active_ros_plugins():
            plugin._load(self, self.ros_node)


    def set_topic_mapping(self, mapping: dict[str, str]):
        def map_fn(name):
            if name in mapping:
                return mapping[name]
            return name + '_sim'
        for plugin in self.ros_plugins:
            plugin.topic_mapping = map_fn



    def compute_footprint(self, alpha: float = 2.0, n=200, base_frame: str = 'base_link'):
        # Polygonalization is NP-hard, so we use alpha shape instead!
        assert self.robot is not None
        
        import alphashape
        import shapely.geometry
        pcd = self.robot.get_pcds(num_points=n)
        assert pcd is not None
        pcd = pcd[..., :2].reshape(-1, 2)

        print("computing alpha shape")
        alpha_shape = alphashape.alphashape(pcd, alpha=alpha)
        assert isinstance(alpha_shape, shapely.geometry.Polygon)

        pcd = np.stack(alpha_shape.exterior.coords.xy, axis=-1)

        frame_pose = self.get_frame(base_frame, self.robot.frame_mapping)[1] # we require the robot defines the frame mapping for the base_link 
        frame_mat = frame_pose.inv().to_transformation_matrix()
        pcd = (np.concatenate([pcd, np.ones((len(pcd), 2))], axis=-1) @ frame_mat.T)[:, :2] # type: ignore

        return {
            'footprint': pcd,
            'radius': np.linalg.norm(pcd, axis=-1).max(),
        }


    def update_footprint(self, g_radius=False, l_radius=False, base_frame: str = 'base_link', **kwargs):
        """_summary_
        compute footprint for ros navigation 
        """
        from std_msgs.msg import Float32
        from geometry_msgs.msg import Polygon, Point32
        import os
        #os.system("ros2 param set /local_costmap/local_costmap robot_radius 0.5")
        footprint = self.compute_footprint(base_frame=base_frame, **kwargs)

        def pub(node_name: str, send_radius):
            node = self.ros_node
            assert node is not None
            if send_radius:
                # def publish():
                os.system(f'ros2 param set /{node_name}/{node_name} robot_radius {footprint["radius"]}')
            else:
                from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
                qos = QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                )
                pub = node.create_publisher(Polygon, f'/{node_name}/footprint', qos)
                pub.publish(Polygon(points=[Point32(x=float(x), y=float(y)) for x, y in footprint['footprint']]))
        pub('local_costmap', l_radius)
        pub('global_costmap', g_radius)



    def get_frame(self, frame: FrameLike, frame_mapping: Optional[dict[str, str]] =None) -> tuple[str, Pose]:
        prev_name = None
        if isinstance(frame, str) and frame_mapping is not None:
            prev_name = frame
            frame = frame_mapping.get(frame, frame)

        if frame == 'world':
            frame_name = 'world'
            pose = Pose()
        elif isinstance(frame, str):
            if frame in self._elem_cache:
                elem = self._elem_cache[frame]
            else:
                elem = self.sim.find(frame)
                self._elem_cache[frame] = elem

            pose = None
            if isinstance(elem, physx.PhysxArticulation):
                pose = elem.get_root_pose()
            elif isinstance(elem, Entity):
                rigid = get_rigid_dynamic_component(elem)
                assert rigid is not None, f"Entity {elem} is not rigid"
                pose = rigid.get_pose()
            elif isinstance(elem, physx.PhysxArticulationLinkComponent):
                pose = elem.get_pose()
            elif isinstance(elem, SensorBase):
                pose = elem.get_pose()
            else:
                raise ValueError(f"Unknown type {type(elem)} for {elem}")

            pose = cast(Pose, pose)
            frame_name = frame
        else:
            return frame(self.sim)

        frame_name = prev_name or frame_name
        return frame_name, pose