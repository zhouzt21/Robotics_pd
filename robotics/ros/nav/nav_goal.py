# https://github.com/haosulab/RealRobot/blob/168e1afd167e15d0f9a8063e7edb79e6b99bce8e/realbot/ros/ros_node.py
from typing import Any
from ..ros_server import ROSServer, ROSNode
from nav2_msgs.action import NavigateToPose
from ..msgs.pose import PoseStamped


def nav_print_feedback(self: ROSServer, feedback):
    self.node.get_logger().info('Feedback: {0}'.format(feedback.feedback))


def nav_done_fn(self: ROSServer, result):
    self.node.get_logger().info('Result: {0}'.format(result.result))

class GoalNavigator(ROSServer):
    cmd = 'ros2 launch ros2_nav nav2.launch.py' 

    def create(self):
        self.goal_navigator = self.create_action_client('navigate_to_pose', NavigateToPose)

        
    def move_to(self, x, y, w, sync=False, timeout=None):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped.to_msg({
            'p': [x, y, 0], 
            'q': [w, 0, 0, 0], 
            'frame': 'map'
        })
        goal.behavior_tree = ''
        return self.send_goal(
            'navigate_to_pose', goal,
            nav_done_fn, nav_print_feedback, sync=sync, timeout=timeout
        )

    def check_map(self):
        # ensure that the map is updated
        pass