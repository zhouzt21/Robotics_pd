from robotics.ros import ROSNode
import time
from robotics.ros.nav.slam import SLAMToolboxConfig, SLAMToolbox

def main():
    node = ROSNode("test_node")
    slam = SLAMToolbox(node, config=SLAMToolboxConfig())

    
    time.sleep(2)
    #slam.save_map('xx')
    slam.serialize_map('xx')
    print('finish serialize')

    slam.deserialize_map('xx', 1)


    node.join()


if __name__ == "__main__":
    main()