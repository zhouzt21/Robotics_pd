# bin
from robotics.ros import ROSNode
from slam_toolbox.srv import SaveMap, SerializePoseGraph

import argparse
parser = argparse.ArgumentParser(description="Run slam, localization and pose")
parser.add_argument("--output", '-o', type=str, default='tmp.pg')
args = parser.parse_args()

node = ROSNode("test")

map_serializer = node.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
map_serializer.wait_for_service()
map_serializer_req = SerializePoseGraph.Request()

map_serializer_req.filename = args.output

output = map_serializer.call(map_serializer_req)
if output.result == 0:
    print("Success")
else:
    print("Failed")