import datajoint as dj
import os
from element_animal import subject
from element_lab import lab
from element_session import session
from element_trial import trial, event
from element_array_ephys import probe

from element_animal.subject import Subject
from element_lab.lab import Source, Lab, Protocol, User, Project
from element_session.session_with_datetime import Session

from .paths import get_ephys_root_data_dir, get_session_directory

if 'custom' not in dj.config:
    dj.config['custom'] = {}

db_prefix = dj.config['custom'].get('database.prefix', '')

# ------------- Import the configured "ephys mode" -------------
ephys_mode = os.getenv('EPHYS_MODE',
                       dj.config['custom'].get('ephys_mode', 'acute'))
if ephys_mode == 'acute':
    from element_array_ephys import ephys
elif ephys_mode == 'chronic':
    from element_array_ephys import ephys_chronic as ephys
elif ephys_mode == 'no-curation':
    from element_array_ephys import ephys_no_curation as ephys
else:
    raise ValueError(f'Unknown ephys mode: {ephys_mode}')
    
__all__ = ['subject', 'lab', 'session', 'trial', 'event', 'probe', 'ephys', 'Subject',
           'Source', 'Lab', 'Protocol', 'User', 'Project', 'Session',
           'get_ephys_root_data_dir', 'get_session_directory']


# Activate "lab", "subject", "session" schema ---------------------------------

lab.activate(db_prefix + 'lab')

subject.activate(db_prefix + 'subject', linking_module=__name__)

Experimenter = lab.User
session.activate(db_prefix + 'session', linking_module=__name__)

trial.activate(db_prefix + 'trial', db_prefix + 'event', linking_module= __name__)


# Declare table "SkullReference" for use in element-array-ephys ---------------

@lab.schema
class SkullReference(dj.Lookup):
    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(['Bregma', 'Lambda'])


# Activate "ephys" schema -----------------------------------------------------

ephys.activate(db_prefix + 'ephys', 
               db_prefix + 'probe', 
               linking_module=__name__)
