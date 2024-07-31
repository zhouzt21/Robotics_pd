from typing import Any, List
from . import ros_api
from rosgraph_msgs.msg import Clock
from .publisher import Publisher, PublisherConfig


class TimePublisher(Publisher):
    def __init__(self, dt, init=None) -> None:
        self.cur = init or 123456
        self.dt = dt
        # ros_api.set_use_sim_time()
        super().__init__(PublisherConfig(channel='/clock', name='time_publisher', queue_size=1), timer=False)

    def get_data_format(self) -> List[Any]:
        return [Clock]

    def fn(self, data: Any) -> Any:
        return data

    def publish(self):
        # Convert nanoseconds to seconds (integer division)
        msg = Clock()
        nanoseconds = self.cur

        seconds = int(nanoseconds / 1000000000)
        nanoseconds = int(nanoseconds % 1000000000)
        msg.clock = ros_api.make_time(secs=seconds, nsecs=nanoseconds)
        super().publish(msg)

        
    def step(self):
        self.cur += self.dt * 1e9
        self.publish()