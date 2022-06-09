import datajoint as dj
import pathlib
import re
import numpy as np
import inspect
import importlib
from element_interface.utils import find_root_directory, find_full_path, dict_to_uuid

from .readers import spikeglx, kilosort, openephys
from . import probe

schema = dj.schema()

_linking_module = None


def activate(ephys_schema_name, probe_schema_name=None, *, create_schema=True,
             create_tables=True, linking_module=None):
    """
    activate(ephys_schema_name, probe_schema_name=None, *, create_schema=True, create_tables=True, linking_module=None)
        :param ephys_schema_name: schema name on the database server to activate the `ephys` element
        :param probe_schema_name: schema name on the database server to activate the `probe` element
         - may be omitted if the `probe` element is already activated
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `ephys` element:
            Upstream tables:
                + Session: parent table to ProbeInsertion, typically identifying a recording session
                + SkullReference: Reference table for InsertionLocation, specifying the skull reference
                 used for probe insertion location (e.g. Bregma, Lambda)
            Functions:
                + get_ephys_root_data_dir() -> list
                    Retrieve the root data directory - e.g. containing the raw ephys recording files for all subject/sessions.
                    :return: a string for full path to the root data directory
                + get_session_directory(session_key: dict) -> str
                    Retrieve the session directory containing the recorded Neuropixels data for a given Session
                    :param session_key: a dictionary of one Session `key`
                    :return: a string for full path to the session directory
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    probe.activate(probe_schema_name, create_schema=create_schema,
                   create_tables=create_tables)
    schema.activate(ephys_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# -------------- Functions required by the elements-ephys  ---------------

def get_ephys_root_data_dir() -> list:
    """
    All data paths, directories in DataJoint Elements are recommended to be 
    stored as relative paths, with respect to some user-configured "root" 
    directory, which varies from machine to machine (e.g. different mounted 
    drive locations)

    get_ephys_root_data_dir() -> list
        This user-provided function retrieves the possible root data directories
         containing the ephys data for all subjects/sessions
         (e.g. acquired SpikeGLX or Open Ephys raw files,
         output files from spike sorting routines, etc.)
        :return: a string for full path to the ephys root data directory,
         or list of strings for possible root data directories
    """
    return _linking_module.get_ephys_root_data_dir()


def get_session_directory(session_key: dict) -> str:
    """
    get_session_directory(session_key: dict) -> str
        Retrieve the session directory containing the
         recorded Neuropixels data for a given Session
        :param session_key: a dictionary of one Session `key`
        :return: a string for relative or full path to the session directory
    """
    return _linking_module.get_session_directory(session_key)


# ----------------------------- Table declarations ----------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """  # Name of software used for recording of neuropixels probes - SpikeGLX or Open Ephys
    acq_software: varchar(24)    
    """
    contents = zip(['SpikeGLX', 'Open Ephys'])


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    # Probe insertion implanted into an animal for a given session.
    -> Session
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """


@schema
class InsertionLocation(dj.Manual):
    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    -> SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysRecording(dj.Imported):
    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion      
    ---
    -> probe.ElectrodeConfig
    -> AcquisitionSoftware
    sampling_rate: float # (Hz)
    recording_datetime: datetime # datetime of the recording from this probe
    recording_duration: float # (seconds) duration of the recording from this probe
    """

    class EphysFile(dj.Part):
        definition = """
        # Paths of files of a given EphysRecording round.
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):

        session_dir = find_full_path(get_ephys_root_data_dir(), 
                                     get_session_directory(key))

        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1('probe')

        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in zip(['*.ap.meta', '*.oebin'],
                                                 ['SpikeGLX', 'Open Ephys']):
            ephys_meta_filepaths = [fp for fp in session_dir.rglob(ephys_pattern)]
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f'Ephys recording data not found!'
                f' Neither SpikeGLX nor Open Ephys recording files found'
                f' in {session_dir}')

        if acq_software == 'SpikeGLX':
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(key))

            if re.search('(1.0|2.0)', spikeglx_meta.probe_model):
                probe_type = spikeglx_meta.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {
                    (shank, shank_col, shank_row): key
                    for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                        'KEY', 'shank', 'shank_col', 'shank_row'))}

                electrode_group_members = [
                    probe_electrodes[(shank, shank_col, shank_row)]
                    for shank, shank_col, shank_row, _ in spikeglx_meta.shankmap['data']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels probe model'
                    ' {} not yet implemented'.format(spikeglx_meta.probe_model))

            self.insert1({**key,
                          **generate_electrode_config(probe_type, electrode_group_members),
                          'acq_software': acq_software,
                          'sampling_rate': spikeglx_meta.meta['imSampRate'],
                          'recording_datetime': spikeglx_meta.recording_time,
                          'recording_duration': (spikeglx_meta.recording_duration
                                        or spikeglx.retrieve_recording_duration(meta_filepath))})

            root_dir = find_root_directory(get_ephys_root_data_dir(), 
                                           meta_filepath)
            self.EphysFile.insert1({
                **key,
                'file_path': meta_filepath.relative_to(root_dir).as_posix()})
        elif acq_software == 'Open Ephys':
            dataset = openephys.OpenEphys(session_dir)
            for serial_number, probe_data in dataset.probes.items():
                if str(serial_number) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No Open Ephys data found for probe insertion: {}'.format(key))

            if re.search('(1.0|2.0)', probe_data.probe_model):
                probe_type = probe_data.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {key['electrode']: key
                                    for key in electrode_query.fetch('KEY')}

                electrode_group_members = [
                    probe_electrodes[channel_idx]
                    for channel_idx in probe_data.ap_meta['channels_ids']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels'
                    ' probe model {} not yet implemented'.format(probe_data.probe_model))

            self.insert1({**key,
                          **generate_electrode_config(probe_type, electrode_group_members),
                          'acq_software': acq_software,
                          'sampling_rate': probe_data.ap_meta['sample_rate'],
                          'recording_datetime': probe_data.recording_info['recording_datetimes'][0],
                          'recording_duration': np.sum(probe_data.recording_info['recording_durations'])})

            root_dir = find_root_directory(get_ephys_root_data_dir(),
                probe_data.recording_info['recording_files'][0])
            self.EphysFile.insert([{**key,
                                    'file_path': fp.relative_to(root_dir).as_posix()}
                                   for fp in probe_data.recording_info['recording_files']])
        else:
            raise NotImplementedError(f'Processing ephys files from'
                                      f' acquisition software of type {acq_software} is'
                                      f' not yet implemented')


@schema
class PreClusterMethod(dj.Lookup):
    definition = """
    # Method for pre-clustering
    precluster_method: varchar(16)
    ---
    precluster_method_desc: varchar(1000)
    """

    contents = [('catgt', 'Time shift, Common average referencing, Zeroing')]


@schema
class PreClusterParamSet(dj.Lookup):
    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> PreClusterMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, precluster_method: str, paramset_idx: int,
                          paramset_desc: str, params: dict):
        param_dict = {'precluster_method': precluster_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(params)}
        param_query = cls & {'param_set_hash': param_dict['param_set_hash']}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1('paramset_idx')
            if existing_paramset_idx == paramset_idx:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    'The specified param-set'
                    ' already exists - paramset_idx: {}'.format(existing_paramset_idx))
        else:
            cls.insert1(param_dict)


@schema
class PreClusterParamSteps(dj.Manual):
    definition = """
    # Ordered list of paramset_idx that are to be run
    # When pre-clustering is not performed, do not create an entry in `Step` Part table
    precluster_param_steps_id: smallint
    ---
    precluster_param_steps_name: varchar(32)
    precluster_param_steps_desc: varchar(128)
    """

    class Step(dj.Part):
        definition = """
        -> master
        step_number: smallint                  # Order of operations
        ---
        -> PreClusterParamSet
        """


@schema
class PreClusterTask(dj.Manual):
    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysRecording
    -> PreClusterParamSteps
    ---
    precluster_output_dir: varchar(255)  #  pre-clustering output directory relative to the root data directory
    task_mode='none': enum('none','load', 'trigger') # 'none': no pre-clustering analysis
                                                     # 'load': load analysis results
                                                     # 'trigger': trigger computation
    """


@schema
class PreCluster(dj.Imported):
    """
    A processing table to handle each PreClusterTask:
    + If `task_mode == "none"`: no pre-clustering performed
    + If `task_mode == "trigger"`: trigger pre-clustering analysis according to the      
                                    PreClusterParamSet
    + If `task_mode == "load"`: verify output
    """
    definition = """
    -> PreClusterTask
    ---
    precluster_time: datetime  # time of generation of this set of pre-clustering results 
    package_version='': varchar(16)
    """

    def make(self, key):
        task_mode, output_dir = (PreClusterTask & key).fetch1('task_mode', 
                                                              'precluster_output_dir')
        precluster_output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == 'none':
            if len((PreClusterParamSteps.Step & key).fetch()) > 0:
                raise ValueError('There are entries in the PreClusterParamSteps.Step '
                                 'table and task_mode=none')
            creation_time = (EphysRecording & key).fetch1('recording_datetime')
        elif task_mode == 'load':
            acq_software = (EphysRecording & key).fetch1('acq_software')
            inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1('probe')

            if acq_software == 'SpikeGLX':
                for meta_filepath in precluster_output_dir.rglob('*.ap.meta'):
                    spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)

                    if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                        creation_time=spikeglx_meta.recording_time
                        break
                else:
                    raise FileNotFoundError(
                        'No SpikeGLX data found for probe insertion: {}'.format(key))
            else:
                raise NotImplementedError(f'Pre-clustering analysis of {acq_software}'
                                          'is not yet supported.')
        elif task_mode == 'trigger':
            raise NotImplementedError('Automatic triggering of'
                                      ' pre-clustering analysis is not yet supported.')
        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        self.insert1({**key, 'precluster_time': creation_time})


@schema
class LFP(dj.Imported):
    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> PreCluster
    ---
    lfp_sampling_rate: float   # (Hz)
    lfp_time_stamps: longblob  # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob         # (uV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode  
        ---
        lfp: longblob               # (uV) recorded lfp at this electrode 
        """

    # Only store LFP for every 9th channel, due to high channel density,
    # close-by channels exhibit highly similar LFP
    _skip_channel_counts = 9

    def make(self, key):
        acq_software, probe_sn = (EphysRecording
                                  * ProbeInsertion & key).fetch1('acq_software', 'probe')

        electrode_keys, lfp = [], []

        if acq_software == 'SpikeGLX':
            spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)

            lfp_channel_ind = spikeglx_recording.lfmeta.recording_channels[
                          -1::-self._skip_channel_counts]

            # Extract LFP data at specified channels and convert to uV
            lfp = spikeglx_recording.lf_timeseries[:, lfp_channel_ind]  # (sample x channel)
            lfp = (lfp * spikeglx_recording.get_channel_bit_volts('lf')[lfp_channel_ind]).T  # (channel x sample)

            self.insert1(dict(key,
                              lfp_sampling_rate=spikeglx_recording.lfmeta.meta['imSampRate'],
                              lfp_time_stamps=(np.arange(lfp.shape[1])
                                               / spikeglx_recording.lfmeta.meta['imSampRate']),
                              lfp_mean=lfp.mean(axis=0)))

            electrode_query = (probe.ProbeType.Electrode
                               * probe.ElectrodeConfig.Electrode
                               * EphysRecording & key)
            probe_electrodes = {
                (shank, shank_col, shank_row): key
                for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                    'KEY', 'shank', 'shank_col', 'shank_row'))}

            for recorded_site in lfp_channel_ind:
                shank, shank_col, shank_row, _ = spikeglx_recording.apmeta.shankmap['data'][recorded_site]
                electrode_keys.append(probe_electrodes[(shank, shank_col, shank_row)])
        elif acq_software == 'Open Ephys':
            
            session_dir = find_full_path(get_ephys_root_data_dir(), 
                                         get_session_directory(key))

            loaded_oe = openephys.OpenEphys(session_dir)
            oe_probe = loaded_oe.probes[probe_sn]

            lfp_channel_ind = np.arange(
                len(oe_probe.lfp_meta['channels_ids']))[-1::-self._skip_channel_counts]

            lfp = oe_probe.lfp_timeseries[:, lfp_channel_ind]  # (sample x channel)
            lfp = (lfp * np.array(oe_probe.lfp_meta['channels_gains'])[lfp_channel_ind]).T  # (channel x sample)
            lfp_timestamps = oe_probe.lfp_timestamps

            self.insert1(dict(key,
                              lfp_sampling_rate=oe_probe.lfp_meta['sample_rate'],
                              lfp_time_stamps=lfp_timestamps,
                              lfp_mean=lfp.mean(axis=0)))

            electrode_query = (probe.ProbeType.Electrode
                               * probe.ElectrodeConfig.Electrode
                               * EphysRecording & key)
            probe_electrodes = {key['electrode']: key
                                for key in electrode_query.fetch('KEY')}

            for channel_idx in np.array(oe_probe.lfp_meta['channels_ids'])[lfp_channel_ind]:
                electrode_keys.append(probe_electrodes[channel_idx])
        else:
            raise NotImplementedError(f'LFP extraction from acquisition software'
                                      f' of type {acq_software} is not yet implemented')

        # single insert in loop to mitigate potential memory issue
        for electrode_key, lfp_trace in zip(electrode_keys, lfp):
            self.Electrode.insert1({**key, **electrode_key, 'lfp': lfp_trace})


# ------------ Clustering --------------

@schema
class ClusteringMethod(dj.Lookup):
    definition = """
    # Method for clustering
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [('kilosort', 'kilosort clustering method'),
                ('kilosort2', 'kilosort2 clustering method')]


@schema
class ClusteringParamSet(dj.Lookup):
    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, processing_method: str, paramset_idx: int,
                          paramset_desc: str, params: dict):
        param_dict = {'clustering_method': processing_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(params)}
        param_query = cls & {'param_set_hash': param_dict['param_set_hash']}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1('paramset_idx')
            if existing_paramset_idx == paramset_idx:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    'The specified param-set'
                    ' already exists - paramset_idx: {}'.format(existing_paramset_idx))
        else:
            cls.insert1(param_dict)


@schema
class ClusterQualityLabel(dj.Lookup):
    definition = """
    # Quality
    cluster_quality_label:  varchar(100)
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ('good', 'single unit'),
        ('ok', 'probably a single unit, but could be contaminated'),
        ('mua', 'multi-unit activity'),
        ('noise', 'bad unit')
    ]


@schema
class ClusteringTask(dj.Manual):
    definition = """
    # Manual table for defining a clustering task ready to be run
    -> PreCluster
    -> ClusteringParamSet
    ---
    clustering_output_dir: varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class Clustering(dj.Imported):
    """
    A processing table to handle each ClusteringTask:
    + If `task_mode == "trigger"`: trigger clustering analysis
        according to the ClusteringParamSet (e.g. launch a kilosort job)
    + If `task_mode == "load"`: verify output
    """
    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results 
    package_version='': varchar(16)
    """

    def make(self, key):
        task_mode, output_dir = (ClusteringTask & key).fetch1(
            'task_mode', 'clustering_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == 'load':
            kilosort_dataset = kilosort.Kilosort(kilosort_dir)  # check if the directory is a valid Kilosort output
            creation_time, _, _ = kilosort.extract_clustering_info(kilosort_dir)
        elif task_mode == 'trigger':
            raise NotImplementedError('Automatic triggering of'
                                      ' clustering analysis is not yet supported')
        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        self.insert1({**key, 'clustering_time': creation_time})


@schema
class Curation(dj.Manual):
    definition = """
    # Manual curation procedure
    -> Clustering
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated clustering results 
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    quality_control: bool               # has this clustering result undergone quality control?
    manual_curation: bool               # has manual curation been performed on this clustering result?
    curation_note='': varchar(2000)  
    """

    def create1_from_clustering_task(self, key, curation_note=''):
        """
        A function to create a new corresponding "Curation" for a particular 
        "ClusteringTask"
        """
        if key not in Clustering():
            raise ValueError(f'No corresponding entry in Clustering available'
                             f' for: {key}; do `Clustering.populate(key)`')

        task_mode, output_dir = (ClusteringTask & key).fetch1(
            'task_mode', 'clustering_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        creation_time, is_curated, is_qc = kilosort.extract_clustering_info(kilosort_dir)
        # Synthesize curation_id
        curation_id = dj.U().aggr(self & key, n='ifnull(max(curation_id)+1,1)').fetch1('n')
        self.insert1({**key, 'curation_id': curation_id,
                      'curation_time': creation_time, 
                      'curation_output_dir': output_dir,
                      'quality_control': is_qc, 
                      'manual_curation': is_curated,
                      'curation_note': curation_note})


@schema
class CuratedClustering(dj.Imported):
    definition = """
    # Clustering results of a curation.
    -> Curation    
    """

    class Unit(dj.Part):
        definition = """   
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording for this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        output_dir = (Curation & key).fetch1('curation_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)
        acq_software = (EphysRecording & key).fetch1('acq_software')

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [i for i, u in enumerate(kilosort_dataset.data['cluster_ids'])
                         if (kilosort_dataset.data['spike_clusters'] == u).any()]
        valid_units = kilosort_dataset.data['cluster_ids'][withspike_idx]
        valid_unit_labels = kilosort_dataset.data['cluster_groups'][withspike_idx]
        # -- Get channel and electrode-site mapping
        channel2electrodes = get_neuropixels_channel2electrode_map(key, acq_software)

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = ('spike_times_sec_adj' if 'spike_times_sec_adj' in kilosort_dataset.data
                          else 'spike_times_sec' if 'spike_times_sec'
                                                    in kilosort_dataset.data else 'spike_times')
        spike_times = kilosort_dataset.data[spike_time_key]
        kilosort_dataset.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = np.array([channel2electrodes[s]['electrode']
                                for s in kilosort_dataset.data['spike_sites']])
        spike_depths = kilosort_dataset.data['spike_depths']

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data['spike_clusters'] == unit).any():
                unit_channel, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (spike_times[kilosort_dataset.data['spike_clusters'] == unit]
                                    / kilosort_dataset.data['params']['sample_rate'])
                spike_count = len(unit_spike_times)

                units.append({
                    'unit': unit,
                    'cluster_quality_label': unit_lbl,
                    **channel2electrodes[unit_channel],
                    'spike_times': unit_spike_times,
                    'spike_count': spike_count,
                    'spike_sites': spike_sites[kilosort_dataset.data['spike_clusters'] == unit],
                    'spike_depths': spike_depths[kilosort_dataset.data['spike_clusters'] == unit] if spike_depths else None})

        self.insert1(key)
        self.Unit.insert([{**key, **u} for u in units])


@schema
class WaveformSet(dj.Imported):
    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """

    def make(self, key):
        output_dir = (Curation & key).fetch1('curation_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)

        acq_software, probe_serial_number = (EphysRecording * ProbeInsertion & key).fetch1(
            'acq_software', 'probe')

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1('KEY')
        channel2electrodes = get_neuropixels_channel2electrode_map(recording_key, acq_software)

        is_qc = (Curation & key).fetch1('quality_control')

        # Get all units
        units = {u['unit']: u for u in (CuratedClustering.Unit & key).fetch(
            as_dict=True, order_by='unit')}

        if is_qc:
            unit_waveforms = np.load(kilosort_dir / 'mean_waveforms.npy')  # unit x channel x sample

            def yield_unit_waveforms():
                for unit_no, unit_waveform in zip(kilosort_dataset.data['cluster_ids'],
                                                  unit_waveforms):
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []
                    if unit_no in units:
                        for channel, channel_waveform in zip(
                                kilosort_dataset.data['channel_map'],
                                unit_waveform):
                            unit_electrode_waveforms.append({
                                **units[unit_no], **channel2electrodes[channel],
                                'waveform_mean': channel_waveform})
                            if channel2electrodes[channel]['electrode'] == units[unit_no]['electrode']:
                                unit_peak_waveform = {
                                    **units[unit_no],
                                    'peak_electrode_waveform': channel_waveform}
                    yield unit_peak_waveform, unit_electrode_waveforms
        else:
            if acq_software == 'SpikeGLX':
                spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                neuropixels_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            elif acq_software == 'Open Ephys':
                session_dir = find_full_path(get_ephys_root_data_dir(), 
                                             get_session_directory(key))
                openephys_dataset = openephys.OpenEphys(session_dir)
                neuropixels_recording = openephys_dataset.probes[probe_serial_number]

            def yield_unit_waveforms():
                for unit_dict in units.values():
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []

                    spikes = unit_dict['spike_times']
                    waveforms = neuropixels_recording.extract_spike_waveforms(
                        spikes, kilosort_dataset.data['channel_map'])  # (sample x channel x spike)
                    waveforms = waveforms.transpose((1, 2, 0))  # (channel x spike x sample)
                    for channel, channel_waveform in zip(
                            kilosort_dataset.data['channel_map'], waveforms):
                        unit_electrode_waveforms.append({
                            **unit_dict, **channel2electrodes[channel],
                            'waveform_mean': channel_waveform.mean(axis=0),
                            'waveforms': channel_waveform})
                        if channel2electrodes[channel]['electrode'] == unit_dict['electrode']:
                            unit_peak_waveform = {
                                **unit_dict,
                                'peak_electrode_waveform': channel_waveform.mean(axis=0)}

                    yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue
        self.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)


# ---------------- HELPER FUNCTIONS ----------------

def get_spikeglx_meta_filepath(ephys_recording_key):
    # attempt to retrieve from EphysRecording.EphysFile
    spikeglx_meta_filepath = (EphysRecording.EphysFile & ephys_recording_key
                              & 'file_path LIKE "%.ap.meta"').fetch1('file_path')

    try:
        spikeglx_meta_filepath = find_full_path(get_ephys_root_data_dir(),
                                                spikeglx_meta_filepath)
    except FileNotFoundError:
        # if not found, search in session_dir again
        if not spikeglx_meta_filepath.exists():
            session_dir = find_full_path(get_ephys_root_data_dir(), 
                                         get_session_directory(
                                             ephys_recording_key))
            inserted_probe_serial_number = (ProbeInsertion * probe.Probe
                                            & ephys_recording_key).fetch1('probe')

            spikeglx_meta_filepaths = [fp for fp in session_dir.rglob('*.ap.meta')]
            for meta_filepath in spikeglx_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    spikeglx_meta_filepath = meta_filepath
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(ephys_recording_key))

    return spikeglx_meta_filepath


def get_neuropixels_channel2electrode_map(ephys_recording_key, acq_software):
    if acq_software == 'SpikeGLX':
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_meta = spikeglx.SpikeGLXMeta(spikeglx_meta_filepath)
        electrode_config_key = (EphysRecording * probe.ElectrodeConfig
                                & ephys_recording_key).fetch1('KEY')

        electrode_query = (probe.ProbeType.Electrode
                           * probe.ElectrodeConfig.Electrode & electrode_config_key)

        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                'KEY', 'shank', 'shank_col', 'shank_row'))}

        channel2electrode_map = {
            recorded_site: probe_electrodes[(shank, shank_col, shank_row)]
            for recorded_site, (shank, shank_col, shank_row, _) in enumerate(
                spikeglx_meta.shankmap['data'])}
    elif acq_software == 'Open Ephys':
        session_dir = find_full_path(get_ephys_root_data_dir(), 
                                     get_session_directory(ephys_recording_key))
        openephys_dataset = openephys.OpenEphys(session_dir)
        probe_serial_number = (ProbeInsertion & ephys_recording_key).fetch1('probe')
        probe_dataset = openephys_dataset.probes[probe_serial_number]

        electrode_query = (probe.ProbeType.Electrode
                           * probe.ElectrodeConfig.Electrode
                           * EphysRecording & ephys_recording_key)

        probe_electrodes = {key['electrode']: key
                            for key in electrode_query.fetch('KEY')}

        channel2electrode_map = {
            channel_idx: probe_electrodes[channel_idx]
            for channel_idx in probe_dataset.ap_meta['channels_ids']}

    return channel2electrode_map


def generate_electrode_config(probe_type: str, electrodes: list):
    """
    Generate and insert new ElectrodeConfig
    :param probe_type: probe type (e.g. neuropixels 2.0 - SS)
    :param electrodes: list of the electrode dict (keys of the probe.ProbeType.Electrode table)
    :return: a dict representing a key of the probe.ElectrodeConfig table
    """
    # compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode)
    electrode_config_hash = dict_to_uuid({k['electrode']: k for k in electrodes})

    electrode_list = sorted([k['electrode'] for k in electrodes])
    electrode_gaps = ([-1]
                      + np.where(np.diff(electrode_list) > 1)[0].tolist()
                      + [len(electrode_list) - 1])
    electrode_config_name = '; '.join([
        f'{electrode_list[start + 1]}-{electrode_list[end]}'
        for start, end in zip(electrode_gaps[:-1], electrode_gaps[1:])])

    electrode_config_key = {'electrode_config_hash': electrode_config_hash}

    # ---- make new ElectrodeConfig if needed ----
    if not probe.ElectrodeConfig & electrode_config_key:
        probe.ElectrodeConfig.insert1({**electrode_config_key, 'probe_type': probe_type,
                                       'electrode_config_name': electrode_config_name})
        probe.ElectrodeConfig.Electrode.insert({**electrode_config_key, **electrode}
                                               for electrode in electrodes)

    return electrode_config_key

