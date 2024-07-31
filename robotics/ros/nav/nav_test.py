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
import os
from robotics.ros import ROSNode
from robotics.ros.nav import slam, localization, nav_goal
import argparse



def main():
    parser = argparse.ArgumentParser(description="Run slam, localization and pose")
    parser.add_argument("--input", '-i', type=str, default=None)
    parser.add_argument("--output", '-o', type=str, default=None)
    args = parser.parse_args()

    node = ROSNode("my_nav_stack")
    slam_server = slam.SLAMToolbox(node, verbose=True)
    navigator = nav_goal.GoalNavigator(node, verbose=False)

    time.sleep(5)
    if args.input is not None:
        assert os.path.exists(args.input + '.posegraph'), f"File {args.input} does not exist"
        node.get_logger().info("Loading map from {}".format(args.input))
        output = slam_server.deserialize_map(args.input, 1, (0, 0, 0))
        node.get_logger().info("Success")

    # try:
    node.join()
    # except KeyboardInterrupt as e:
    #     if args.output is not None:
    #         node.get_logger().info("Saving map to {}".format(args.output))
    #         slam_server.serialize_map(args.output)
    #     print("Exiting...")
    #     raise e


if __name__ == '__main__':
    main()