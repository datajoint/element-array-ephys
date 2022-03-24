import datajoint as dj
from element_electrode_localization import coordinate_framework, electrode_localization
from element_electrode_localization.coordinate_framework import load_ccf_annotation

from .pipeline import ephys, probe
from .paths import get_ephys_root_data_dir, get_session_directory, \
                   get_electrode_localization_dir


if 'custom' not in dj.config:
    dj.config['custom'] = {}

db_prefix = dj.config['custom'].get('database.prefix', '')

__all__ = ['ephys', 'probe', 'coordinate_framework', 'electrode_localization',
           'ProbeInsertion',
           'get_ephys_root_data_dir', 'get_session_directory',
           'get_electrode_localization_dir', 'load_ccf_annotation']


# Dummy table for case sensitivity in MySQL------------------------------------

coordinate_framework_schema = dj.schema(db_prefix + 'ccf')

@coordinate_framework_schema
class DummyTable(dj.Manual):
    definition = """
    id   : varchar(1)
    """
    contents = zip(['1', '2'])


ccf_schema_name = db_prefix + 'ccf'
dj.conn().query(f'ALTER DATABASE `{ccf_schema_name}` CHARACTER SET utf8 COLLATE '
                + 'utf8_bin;')



# Activate "electrode-localization" schema ------------------------------------

ProbeInsertion = ephys.ProbeInsertion
Electrode = probe.ProbeType.Electrode
electrode_localization.activate(db_prefix + 'eloc',
                                db_prefix + 'ccf',
                                linking_module=__name__)
