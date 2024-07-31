
import numpy as np
import time
from robotics.ros import ROSNode, Publisher, rtypes
from robotics.ros import rtypes

from example_interfaces.srv import AddTwoInts

class Input(rtypes.Struct):
    a: int
    b: int

class Out(rtypes.Struct):
    sum: int

class AddTwoIntsService(rtypes.ServiceFormat):
    dtype = AddTwoInts
    inp = Input
    out = Out

def main():
    node = ROSNode("test_node")
    import tqdm

    client = node.create_client("/add_two_ints", AddTwoIntsService)
    print(client(a=1, b=2))
    print('123')


if __name__ == "__main__":
    main()