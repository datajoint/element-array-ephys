import logging
import os

import datajoint as dj

from . import ephys_acute as ephys

__all__ = ["ephys", "get_logger"]

dj.config["enable_python_native_blobs"] = True


def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(os.getenv("LOGLEVEL", "INFO"))
    return log
