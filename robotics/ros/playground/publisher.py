from robotics.utils import Config
from typing import Any, Optional, List, Tuple, Callable
from torch.multiprocessing import Process, Pipe


class PublisherConfig(Config):
    name: str = 'publisher'
    channel: str = ''
    queue_size: Optional[int] = 10
    rate: float = 10.
    cmd: Optional[str] = None


class Publisher(Process):
    data_format: List[Any]
    config: PublisherConfig

    def __init__(self, config: PublisherConfig, timer=True) -> None:
        super().__init__()
        self.config = config
        self.parent, self.child = Pipe()
        self.start()

    def fn(self, data: Any) -> Any:
        raise NotImplementedError

    def get_data_format(self) -> List[Any]:
        raise NotImplementedError

    def run(self) -> None:
        print("publisher start..")
        from . import ros_api
        node = ros_api.init(self.config.name)
        self.data_format = self.get_data_format()

        channels = []
        for idx, channel in enumerate(self.config.channel.split(',')):
            channels.append(ros_api.create_publisher(channel, self.data_format[idx], queue_size=self.config.queue_size))

        _data = None

        def sender():
            nonlocal _data
            data = self.fn(_data) # call node.get_clock().now() inside fn
            assert len(data) == len(channels), f"{len(data)} != {len(channels)}"

            for k, v in enumerate(data):
                if v is not None:
                    channels[k].publish(v)

        node.create_timer(0., sender)
        while not ros_api.is_shutdown():
            _data = self.child.recv()
            if _data == 'EXIT':
                break

            # sender() # can not be called outside the spin once
            ros_api.spin_once()


        print("publisher exit..")
        ros_api.close()


    def close(self) -> None:
        self.parent.send('EXIT')

    def publish(self, *data: Any) -> None:
        self.parent.send(data)

    def __del__(self):
        print('stop publisher')
        self.close()
