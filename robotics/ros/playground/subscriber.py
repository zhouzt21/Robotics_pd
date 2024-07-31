from robotics.utils import Config
from typing import Any, Optional, List, Tuple, Callable
from torch.multiprocessing import Process, Pipe


class SubscriberConfig(Config):
    name: str = 'subscriber'
    channel: str = ''
    queue_size: Optional[int] = 10


class _Subscriber(Process):
    data_format: List[Any]
    config: SubscriberConfig

    def __init__(self, config: SubscriberConfig, data_format_fn, fn) -> None:
        super().__init__()
        self.config = config
        self.data_format = data_format_fn()
        self.fn = fn
        self.parent, self.child = Pipe()

    def run(self) -> None:
        print("subscriber start..")
        from . import ros_api
        node = ros_api.init(self.config.name)
        def fn(data):
            if self.child.poll():
                ros_api.close()
            self.child.send(self.fn(data))
        self.subscriber = ros_api.create_subscriber(self.config.channel, self.data_format, self.config.queue_size, fn)
        ros_api.spin()


class Subscriber:
    # ensure that the worker will be closed ..
    def __init__(self, config: SubscriberConfig) -> None:
        super().__init__()
        _subprocess = _Subscriber(config, *self.get_fn())
        _subprocess.daemon = True
        self.pipe = _subprocess.parent
        _subprocess.start()

    def get_fn(self) -> Tuple[Callable[[], List[Any]], Callable]:
        return lambda x: [], lambda x: x

    def close(self) -> None:
        self.pipe.send('EXIT')

    def __del__(self):
        print('stop publisher')
        self.close()


class NavCmdVelSubscriber(Subscriber):
    def get_fn(self) -> Tuple[Callable[[], List[Any]], Callable]:
        def get():
            from geometry_msgs.msg import Twist
            return Twist 
        return get, lambda x: x

    def poll(self) -> Any:
        return self.pipe.poll()

    def recv(self) -> Any:
        return self.pipe.recv()