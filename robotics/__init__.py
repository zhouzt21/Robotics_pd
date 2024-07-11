import os
from typing import Dict, Union, List, TYPE_CHECKING, Sequence, overload, Tuple

PATH = os.path.dirname(os.path.abspath(__file__))



if TYPE_CHECKING:
    #NOTE: a temporary solution to pass type checking
    import numpy as np
    from sapien.core import Pose as PoseBase
    class Pose(PoseBase):
        def __init__(self, p: np.ndarray|Sequence[float|int]=(0, 0, 0), q: np.ndarray |Sequence[float|int]=(1, 0, 0, 0)):
            pass
else:
    from sapien.core import Pose