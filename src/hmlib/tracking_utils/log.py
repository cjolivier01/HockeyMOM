import logging
import sys


def get_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(sys.stderr)
    # handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    #logger.setLevel(logging.INFO)

    logger.addHandler(handler)
    return logger


logger = get_logger('root')
