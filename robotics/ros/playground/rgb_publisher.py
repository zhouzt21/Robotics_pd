from .publisher import Any, List, Publisher, PublisherConfig, Optional, Tuple
from .nav_sensor_publisher import pose2msg, create_tf_msgs
from . import ros_api
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import numpy as np

bridge = CvBridge()


def matrix4x4tomsg(matrix):
    matrix_msg = Float64MultiArray()
    matrix_msg.layout.dim.append(MultiArrayDimension())
    matrix_msg.layout.dim[0].label = "rows"
    matrix_msg.layout.dim[0].size = 4
    matrix_msg.layout.dim[0].stride = 16
    matrix_msg.layout.dim.append(MultiArrayDimension())
    matrix_msg.layout.dim[1].label = "columns"
    matrix_msg.layout.dim[1].size = 4
    matrix_msg.layout.dim[1].stride = 4
    matrix_msg.layout.data_offset = 0
    flat_matrix = [float(num) for row in matrix for num in row]
    matrix_msg.data = flat_matrix
    return matrix_msg

def matrix_msg2numpy(matrix_msg: Float64MultiArray):
    matrix = np.array(matrix_msg.data).reshape((4, 4))
    return matrix


class RGBPublisherConfig(PublisherConfig):
    channel: str = '/rgbd,/camera_matrix,/intrinsic'
    camera: str = 'base_sensor'

    
class RGBPublisher(Publisher):
    config: RGBPublisherConfig

    def get_data_format(self) -> List[Any]:
        from sensor_msgs.msg import Image
        return [Image, Float64MultiArray, Float64MultiArray]

    def fn(self, data_: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Image, Float64MultiArray, Float64MultiArray]:

        data, depth, matrix, intrinsic = data_

        assert data.dtype == np.uint8, f"{data.dtype} is not uint8. {data}"

        _intrinsic = np.zeros((4, 4))
        _intrinsic[:3, :3] = intrinsic

        rgbd = np.concatenate((np.float32(data/255.), depth[..., None]), axis=2)

        image = bridge.cv2_to_imgmsg(rgbd, encoding='32FC4')

        image.header.stamp = ros_api.get_time()
        image.header.frame_id = self.config.camera
        return (image, matrix4x4tomsg(matrix), matrix4x4tomsg(_intrinsic))