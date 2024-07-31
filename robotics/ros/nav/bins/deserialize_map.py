# bin
from robotics.ros import ROSNode
from slam_toolbox.srv import SaveMap, DeserializePoseGraph

import argparse
parser = argparse.ArgumentParser(description="Run slam, localization and pose")
parser.add_argument("--input", '-i', type=str, default='tmp.pg')
args = parser.parse_args()

node = ROSNode("test")

map_deserializer = node.create_client(DeserializePoseGraph, '/slam_toolbox/deserialize_map')
map_deserializer.wait_for_service()
map_deserializer_req = DeserializePoseGraph.Request()

map_deserializer_req.filename = args.input
map_deserializer_req.match_type = 1
map_deserializer_req.initial_pose.x = 0.0
map_deserializer_req.initial_pose.y = 0.0
map_deserializer_req.initial_pose.theta = 0.0

output = map_deserializer.call(map_deserializer_req)
if output.result == 0:
    print("Success")
else:
    print("Failed")