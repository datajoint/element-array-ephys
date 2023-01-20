"""
isort:skip_file
"""

import logging
import os

import datajoint as dj


__all__ = ["ephys", "get_logger"]

dj.config["enable_python_native_blobs"] = True


def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(os.getenv("LOGLEVEL", "INFO"))
    return log


from . import ephys_acute as ephys
