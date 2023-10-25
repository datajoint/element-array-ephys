import datajoint as dj
from element_animal import subject
from element_animal.subject import Subject
from element_array_ephys import db_prefix, probe, ephys_acute as ephys
from element_lab import lab
from element_lab.lab import Lab, Location, Project, Protocol, Source, User
from element_lab.lab import Device as Equipment
from element_lab.lab import User as Experimenter
from element_session import session_with_id as session
from element_session.session_with_id import Session
import element_interface
import pathlib


# Declare functions for retrieving data
def get_ephys_root_data_dir():
    """Retrieve ephys root data directory."""
    ephys_root_dirs = dj.config.get("custom", {}).get("ephys_root_data_dir", None)
    if not ephys_root_dirs:
        return None
    elif isinstance(ephys_root_dirs, (str, pathlib.Path)):
        return [ephys_root_dirs]
    elif isinstance(ephys_root_dirs, list):
        return ephys_root_dirs
    else:
        raise TypeError("`ephys_root_data_dir` must be a string, pathlib, or list")

# Activate schemas
lab.activate(db_prefix + "lab")
subject.activate(db_prefix + "subject", linking_module=__name__)
session.activate(db_prefix + "session", linking_module=__name__)
ephys.activate(db_prefix + "ephys", db_prefix + "probe", linking_module=__name__)
