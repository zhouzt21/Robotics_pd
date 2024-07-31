"""
ROS Node (threading version). We open one single thread to spining publishers, clients, subcribers and so on.
It is still in development, and we are not sure if it is the best way to do it.
See the example in  tester/test_ros_node.py, tester/test_client.py (for client) and nav/slam.py for more details.


# https://discourse.ros.org/t/how-to-use-callback-groups-in-ros2/25255
"""
import os

import rclpy
import threading
import subprocess
from typing import Any, Optional, List, Callable
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy


def get_ros2_node_names():
    # Run the 'ros2 node list' command
    result = subprocess.run(['ros2', 'node', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was executed successfully
    if result.returncode == 0:
        # Split the output by newlines to get individual node names
        node_names = result.stdout.strip().split('\n')
        return node_names
    else:
        # If there was an error, print it and return an empty list
        print(f"Error: {result.stderr}")
        return []



class ROSNode(Node):
    def __init__(self, name: str, use_sim_time: bool = True, args: Optional[List[Any]]=None, ros_domain_id: Optional[int] = None):
        if ros_domain_id is not None:
            os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)

        rclpy.init(args=args)

        super().__init__(name) # type: ignore 
        self.name = name
        if name in get_ros2_node_names():
            raise ValueError(f"Node {name} already exists")

        self.use_sim_time = use_sim_time
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, self.use_sim_time)])

        self.thread = threading.Thread(target=rclpy.spin, daemon=True, args=(self,))
        self.start()

        self._static_broadcaster = None


    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    @property
    def static_broadcaster(self):
        from tf2_ros import StaticTransformBroadcaster
        if self._static_broadcaster is None:
            self._static_broadcaster = StaticTransformBroadcaster(self)
        return self._static_broadcaster


    def sleep(self, t: float):
        import time
        time.sleep(t)

    def close(self):
        rclpy.shutdown()
        self.join()

    def __del__(self):
        if rclpy.ok():
            self.close()

    def listen_once(self, msg_type, topic: str, callback: Callable, timeout: Optional[float] = 10., is_static: bool=True):
        print('listen_once', topic)
        start_event = threading.Event()
        end_event = threading.Event()
        if is_static:
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        else:
            qos_profile = 1

        def callback_once(msg):
            nonlocal end_event, start_event
            start_event.wait()
            callback(msg)
            self.destroy_subscription(subcription)
            end_event.set()

        subcription = self.create_subscription(msg_type, topic, callback_once, qos_profile)
        start_event.set()
        end_event.wait(timeout)
        if not end_event.is_set():
            raise TimeoutError(f"Timeout when listening to {topic}")

    def publish_once(self, msg_type, topic: str, msg):
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publisher = self.create_publisher(msg_type, topic, qos_profile)
        publisher.publish(msg)
        self.destroy_publisher(publisher)