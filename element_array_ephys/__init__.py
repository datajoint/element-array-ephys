import datajoint as dj
import logging
import os


dj.config['enable_python_native_blobs'] = True


def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(os.getenv('LOGLEVEL', 'INFO'))
    return log