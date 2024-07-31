from typing import Any, Optional
import numpy as np
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge

bridge = CvBridge()


class Image:
    dtype = ImageMsg
    @classmethod
    def from_msg(cls, msg: ImageMsg):
        encoding = msg.encoding
        dim = int(encoding[-1])
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width, dim))

        
    @classmethod
    def to_msg(cls, img, frame: Optional[str]=None):
        msg =  bridge.cv2_to_imgmsg(img, encoding=f'32FC{img.shape[-1]}')
        if frame is not None:
            msg.header.frame_id = frame
        return msg