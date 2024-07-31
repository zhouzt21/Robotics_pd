import numpy as np
import time
from robotics.ros import ROSNode, Publisher, rtypes


def main():
    node = ROSNode("test_node")
    publisher = node.create_publisher("/image", rtypes.Image, 10)
    subscriber = node.create_subscriber("/image", rtypes.Image, 10)


    image = np.random.random((512, 512, 3)).astype(np.float32)

    publisher.publish(image)

    node.sleep(0.1) # initialize

    out = subscriber.get()
    assert np.allclose(image, out)

    node.close()


if __name__ == "__main__":
    main()