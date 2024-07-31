from .ros_node import ROSNode
import os
import sys
import subprocess
from typing import Optional
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.task import Future
from typing import Any, Callable, Dict
import threading


class ROSServer:
    """_summary_
    ROS services: It launches a series of ROS2 package in the backend, and provides a series of APIs to interact with them. This helps us to abstract the details in the ROS2 package and focus on the python side. 

    A ROS2 node is required. Then ROSServer can create clients, publishers and subscribers for communicating with the ROS2 package.

    Note we do not allow multiple ROS2 package launched simultaneously right now, thus we design it as a singleton.
    """
    cmd: str
    SINGLETON = None
    config: Optional[dict] = None
    node_name: str

    def __init__(self, node: ROSNode, cmd: Optional[str]=None, verbose: bool=True) -> None:
        if self.__class__.SINGLETON is not None:
            raise Exception("This class is a singleton!")
        self.__class__.SINGLETON = self

        self.node = node
        self.cmd = cmd or self.cmd
        self.verbose = verbose
        self.action_clients: Dict[str, ActionClient] = {}
        self.clients: Dict[str, Client] = {}

        self.start()

    def create_action_client(self, topic: str, msg_type: Any):
        self.action_clients[topic] = ActionClient(self.node, msg_type, topic)
        return self.action_clients[topic]

    def create_client(self, service_name: str, msg_type: Any):
        self.node.get_logger().info(f"Creating client for service {service_name}")
        client = self.node.create_client(msg_type, service_name)
        self.clients[service_name] = client
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f'service {service_name} not available, waiting again...')
        return client

        
    def start(self):
        self.start_process()
        self.create()

    def stop(self):
        self.process.terminate()

    def create(self):
        """_summary_
        create clients, publishers and subscribers here.
        """
        pass
        
    def start_process(self):
        """_summary_
        laucnh the ROS2 package in the backend. If config is not None, it will be passed to the command line as arguments. 
        """
        cmd = self.cmd
        if self.config is not None:
            cmd = cmd + " " + " ".join([f"{k}:={v}" for k, v in self.config.items() if v is not None])
            
        if self.verbose:
            self.process = subprocess.Popen(cmd.split(' '), stdout=sys.stdout, stderr=sys.stderr)
        else:
            self.process = subprocess.Popen(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def set_parameters(self, params: dict):
        assert hasattr(self, 'node_name'), "You must specify the node_name attribute!"
        for k, (t, v) in params.items():
            #self.node.set_parameters({k: v})
            os.system(f"ros2 param set {self.node_name} {k} {v}")

            

    def send_goal(
        self, topic, goal_msg, done_fn: Callable, 
        feedback_fn: Optional[Callable] = None,
        sync=True, 
        timeout = None,
    ):
        action_client: ActionClient = self.action_clients[topic]
        while not action_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().info(f'Action {topic} not available, waiting again...')

        ret_result = None
        event = threading.Event() if sync else None

        def get_result_callback(future: Future):
            nonlocal ret_result, event
            result = future.result()
            done_fn(self, result)
            ret_result = result
            if event is not None:
                event.set()


        def done_callback(future: Future):
            nonlocal event, ret_result
            goal_handle: Any = future.result() # TODO: figure out the type of the goal handle
            if not goal_handle.accepted:
                self.node.get_logger().info('Goal rejected :(')
                ret_result = 'Failed'
                if event is not None:
                    event.set()
                return
            self.node.get_logger().info('Goal accepted :)')
            _get_result_future = goal_handle.get_result_async()
            _get_result_future.add_done_callback(get_result_callback)

        def feedback_callback(feedback_msg):
            if feedback_fn is not None:
                feedback_fn(self, feedback_msg)

        action_future = action_client.send_goal_async(goal_msg, feedback_callback=feedback_callback)
        action_future.add_done_callback(done_callback)

        if event is not None:
            if not action_future.done():
                event.wait(timeout=timeout)

            if timeout is not None and not action_future.done():
                action_future.cancel()

            exception = action_future.exception()
            if exception is not None:
                raise exception
            return ret_result
        else:
            return action_future
    
    def call_service(self, service_name: str, request: Any):
        client = self.clients[service_name]
        return client.call(request)
        
        
class Recorder(ROSServer):
    cmd = 'ros2 bag record -a'