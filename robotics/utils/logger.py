import time
import pickle
import logging
import contextlib

# 0 system
# 1 info
# 2 debug


class Logger:
    def __init__(self, level, verbose=False, dump=True) -> None:
        self.message = []
        self.level = level
        self.verbose = verbose
        self.dump = dump

    def log(self, key, value, level=1):
        # log only when level is smaller than message
        if level > self.level:
            return
        
        if self.verbose:
            print(key, value)
        
        self.message.append((key, value, time.time()))


    def close(self):
        if len(self.message) > 0 and self.dump:
            with open("log.pkl", "wb") as f:
                pickle.dump(self.message, f)

    def animate(self, key, filename='test.mp4', fps=30):
        images = [i[1] for i in self.message if i[0] == key]
        from .utils import animate
        animate(images, filename, fps=fps)


logger = None

@contextlib.contextmanager
def configure(level, **kwargs):
    global logger
    logger = Logger(level, **kwargs)
    yield logger
    logger.close()


def log(key, value, level=1):
    global logger
    if logger is None:
        logging.warning("Logger is not configured")
    else:
        logger.log(key, value, level)