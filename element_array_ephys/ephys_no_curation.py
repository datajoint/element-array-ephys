import datajoint as dj
import pathlib
import re
import numpy as np
import inspect
import importlib
import gc
from decimal import Decimal
import os

from element_interface.utils import find_root_directory, find_full_path, dict_to_uuid

from .readers import spikeglx, kilosort, openephys
from . import probe, get_logger

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import spikeinterface.preprocessing as sip
import probeinterface as pi

log = get_logger(__name__)

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
                + Session: table referenced by EphysRecording, typically identifying a recording session
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
                + get_processed_root_data_dir() -> str:
                    Retrieves the root directory for all processed data to be found from or written to
                    :return: a string for full path to the root directory for processed data
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    # activate
    probe.activate(probe_schema_name, create_schema=create_schema,
                   create_tables=create_tables)
    schema.activate(ephys_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# -------------- Functions required by the elements-ephys  ---------------

def get_ephys_root_data_dir() -> list:
    """
    All data paths, directories in DataJoint Elements are recommended to be stored as
    relative paths, with respect to some user-configured "root" directory,
     which varies from machine to machine (e.g. different mounted drive locations)

    get_ephys_root_data_dir() -> list
        This user-provided function retrieves the possible root data directories
         containing the ephys data for all subjects/sessions
         (e.g. acquired SpikeGLX or Open Ephys raw files,
         output files from spike sorting routines, etc.)
        :return: a string for full path to the ephys root data directory,
         or list of strings for possible root data directories
    """
    root_directories = _linking_module.get_ephys_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, 'get_processed_root_data_dir'):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_session_directory(session_key: dict) -> str:
    """
    get_session_directory(session_key: dict) -> str
        Retrieve the session directory containing the
         recorded Neuropixels data for a given Session
        :param session_key: a dictionary of one Session `key`
        :return: a string for full path to the session directory
    """
    return _linking_module.get_session_directory(session_key)


def get_processed_root_data_dir() -> str:
    """
    get_processed_root_data_dir() -> str:
        Retrieves the root directory for all processed data to be found from or written to
        :return: a string for full path to the root directory for processed data
    """

    if hasattr(_linking_module, 'get_processed_root_data_dir'):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_ephys_root_data_dir()[0]

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

    @classmethod
    def auto_generate_entries(cls, session_key):
        """
        Method to auto-generate ProbeInsertion entries for a particular session
        Probe information is inferred from the meta data found in the session data directory
        """
        session_dir = find_full_path(get_ephys_root_data_dir(),
                                  get_session_directory(session_key))
        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in zip(['*.ap.meta', '*.oebin'],
                                                 ['SpikeGLX', 'Open Ephys']):
            ephys_meta_filepaths = list(session_dir.rglob(ephys_pattern))
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f'Ephys recording data not found!'
                f' Neither SpikeGLX nor Open Ephys recording files found in: {session_dir}')

        probe_list, probe_insertion_list = [], []
        if acq_software == 'SpikeGLX':
            for meta_fp_idx, meta_filepath in enumerate(ephys_meta_filepaths):
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)

                probe_key = {'probe_type': spikeglx_meta.probe_model,
                             'probe': spikeglx_meta.probe_SN}
                if (probe_key['probe'] not in [p['probe'] for p in probe_list]
                        and probe_key not in probe.Probe()):
                    probe_list.append(probe_key)

                probe_dir = meta_filepath.parent
                try:
                    probe_number = re.search('(imec)?\d{1}$', probe_dir.name).group()
                    probe_number = int(probe_number.replace('imec', ''))
                except AttributeError:
                    probe_number = meta_fp_idx

                probe_insertion_list.append({**session_key,
                                             'probe': spikeglx_meta.probe_SN,
                                             'insertion_number': int(probe_number)})
        elif acq_software == 'Open Ephys':
            loaded_oe = openephys.OpenEphys(session_dir)
            for probe_idx, oe_probe in enumerate(loaded_oe.probes.values()):
                probe_key = {'probe_type': oe_probe.probe_model, 'probe': oe_probe.probe_SN}
                if (probe_key['probe'] not in [p['probe'] for p in probe_list]
                        and probe_key not in probe.Probe()):
                    probe_list.append(probe_key)
                probe_insertion_list.append({**session_key,
                                             'probe': oe_probe.probe_SN,
                                             'insertion_number': probe_idx})
        else:
            raise NotImplementedError(f'Unknown acquisition software: {acq_software}')

        probe.Probe.insert(probe_list)
        cls.insert(probe_insertion_list, skip_duplicates=True)


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
            ephys_meta_filepaths = list(session_dir.rglob(ephys_pattern))
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f'Ephys recording data not found!'
                f' Neither SpikeGLX nor Open Ephys recording files found')

        supported_probe_types = probe.ProbeType.fetch('probe_type')

        if acq_software == 'SpikeGLX':
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(key))

            if spikeglx_meta.probe_model in supported_probe_types:
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

            self.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': spikeglx_meta.meta['imSampRate'],
                'recording_datetime': spikeglx_meta.recording_time,
                'recording_duration': (spikeglx_meta.recording_duration
                                       or spikeglx.retrieve_recording_duration(meta_filepath))})

            root_dir = find_root_directory(get_ephys_root_data_dir(), meta_filepath)
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

            if not probe_data.ap_meta:
                raise IOError('No analog signals found - check "structure.oebin" file or "continuous" directory')

            if probe_data.probe_model in supported_probe_types:
                probe_type = probe_data.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {key['electrode']: key
                                    for key in electrode_query.fetch('KEY')}

                electrode_group_members = [
                    probe_electrodes[channel_idx]
                    for channel_idx in probe_data.ap_meta['channels_indices']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels'
                    ' probe model {} not yet implemented'.format(probe_data.probe_model))

            self.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': probe_data.ap_meta['sample_rate'],
                'recording_datetime': probe_data.recording_info['recording_datetimes'][0],
                'recording_duration': np.sum(probe_data.recording_info['recording_durations'])})

            root_dir = find_root_directory(
                get_ephys_root_data_dir(),
                probe_data.recording_info['recording_files'][0])
            self.EphysFile.insert([{**key,
                                    'file_path': fp.relative_to(root_dir).as_posix()}
                                   for fp in probe_data.recording_info['recording_files']])
            # explicitly garbage collect "dataset"
            # as these may have large memory footprint and may not be cleared fast enough
            del probe_data, dataset
            gc.collect()
        else:
            raise NotImplementedError(f'Processing ephys files from'
                                      f' acquisition software of type {acq_software} is'
                                      f' not yet implemented')


@schema
class LFP(dj.Imported):
    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> EphysRecording
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
        acq_software = (EphysRecording * ProbeInsertion & key).fetch1('acq_software')

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
            oe_probe = get_openephys_probe_data(key)

            lfp_channel_ind = np.r_[
                len(oe_probe.lfp_meta['channels_indices'])-1:0:-self._skip_channel_counts]

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

            electrode_keys.extend(probe_electrodes[channel_idx]
                                  for channel_idx in lfp_channel_ind)
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

    contents = [('kilosort2', 'kilosort2 clustering method'),
                ('kilosort2.5', 'kilosort2.5 clustering method'),
                ('kilosort3', 'kilosort3 clustering method')]


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
    def insert_new_params(cls, clustering_method: str, paramset_desc: str,
                          params: dict, paramset_idx: int = None):
        if paramset_idx is None:
            paramset_idx = (dj.U().aggr(cls, n='max(paramset_idx)').fetch1('n') or 0) + 1

        param_dict = {'clustering_method': clustering_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(
                          {**params, 'clustering_method': clustering_method})
                      }
        param_query = cls & {'param_set_hash': param_dict['param_set_hash']}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1('paramset_idx')
            if existing_paramset_idx == paramset_idx:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f'The specified param-set already exists'
                    f' - with paramset_idx: {existing_paramset_idx}')
        else:
            if {'paramset_idx': paramset_idx} in cls.proj():
                raise dj.DataJointError(
                    f'The specified paramset_idx {paramset_idx} already exists,'
                    f' please pick a different one.')
            cls.insert1(param_dict)


@schema
class ClusterQualityLabel(dj.Lookup):
    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
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
    -> EphysRecording
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """
        Given a 'key' to an entry in this table
        Return the expected clustering_output_dir based on the following convention:
            processed_dir / session_dir / probe_{insertion_number} / {clustering_method}_{paramset_idx}
            e.g.: sub4/sess1/probe_2/kilosort2_0
        """
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        session_dir = find_full_path(get_ephys_root_data_dir(),
                                  get_session_directory(key))
        root_dir = find_root_directory(get_ephys_root_data_dir(), session_dir)

        method = (ClusteringParamSet * ClusteringMethod & key).fetch1(
            'clustering_method').replace(".", "-")

        output_dir = (processed_dir
                      / session_dir.relative_to(root_dir)
                      / f'probe_{key["insertion_number"]}'
                      / f'{method}_{key["paramset_idx"]}')

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f'{output_dir} created!')

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, ephys_recording_key, paramset_idx=0):
        """
        Method to auto-generate ClusteringTask entries for a particular ephys recording
            Output directory is auto-generated based on the convention
             defined in `ClusteringTask.infer_output_dir()`
            Default parameter set used: paramset_idx = 0
        """
        key = {**ephys_recording_key, 'paramset_idx': paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = ClusteringTask.infer_output_dir(key, relative=False, mkdir=True)

        try:
            kilosort.Kilosort(output_dir)  # check if the directory is a valid Kilosort output
        except FileNotFoundError:
            task_mode = 'trigger'
        else:
            task_mode = 'load'

        cls.insert1({
            **key,
            'clustering_output_dir': output_dir.relative_to(processed_dir).as_posix(),
            'task_mode': task_mode})


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

        if not output_dir:
            output_dir = ClusteringTask.infer_output_dir(key, relative=True, mkdir=True)
            # update clustering_output_dir
            ClusteringTask.update1({**key, 'clustering_output_dir': output_dir.as_posix()})

        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == 'load':
            kilosort.Kilosort(kilosort_dir)  # check if the directory is a valid Kilosort output
        elif task_mode == 'trigger':
            acq_software, clustering_method, params = (ClusteringTask * EphysRecording
                                                       * ClusteringParamSet & key).fetch1(
                'acq_software', 'clustering_method', 'params')

            if 'kilosort' in clustering_method:
                # from element_array_ephys.readers import kilosort_triggering

                # add additional probe-recording and channels details into `params`
                params = {**params, **get_recording_channels_details(key)}
                params['fs'] = params['sample_rate']

                if acq_software == 'SpikeGLX':
                    
                    spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                    spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
                    spikeglx_recording.validate_file('ap')

                    stream_name = os.path.split(spikeglx_meta_filepath)[1]
                    
                    oe_si_recording = se.OpenEphysBinaryRecordingExtractor(folder_path=spikeglx_meta_filepath, stream_name=stream_name) 

                    sglx_si_recording = se.SpikeGLXRecordingExtractor(spikeglx_meta_filepath,"imec.ap")

                    electrode_query = (probe.ProbeType.Electrode
                                    * probe.ElectrodeConfig.Electrode
                                    * EphysRecording & key)

                    xy_coords = [list(i) for i in zip(electrode_query.fetch('x_coord'), electrode_query.fetch('y_coord'))]

                    channel_details = get_recording_channels_details(key)
                        

                    probe = pi.Probe(ndim=2, si_units='um')
                    probe.set_contacts(positions=xy_coords, shapes='square', shape_params={'width': 5})
                    probe.create_auto_shape(probe_type='tip')


                    channel_indices = np.arange(channel_details['num_channels'])
                    probe.set_device_channel_indices(channel_indices)
                    sglx_si_recording.set_probe(probe=probe)
 

                    # SpikeInterface Recording Object Preprocessing 
                    # Apply bandpass filter and rewrite recording object
                    # Store common mode referenced recording for construction of waveform extractor object
                    recording_cmr = sglx_si_recording
                    oe_si_recording = sip.bandpass_filter(sglx_si_recording, freq_min=300, freq_max=6000)
                    recording_cmr = sip.common_reference(sglx_si_recording, reference="global", operator="median")

                    """ Preprocessing Options - Can choose optimal pipeline
                        'bandpass_filter': <class 'spikeinterface.preprocessing.filter.BandpassFilterRecording'>,
                        'blank_staturation': <class 'spikeinterface.preprocessing.clip.BlankSaturationRecording'>,
                        'center': <class 'spikeinterface.preprocessing.normalize_scale.ZScoreRecording'>,
                        'clip': <class 'spikeinterface.preprocessing.clip.ClipRecording'>,
                        'common_reference': <class 'spikeinterface.preprocessing.common_reference.CommonReferenceRecording'>,
                        'deepinterpolate': <class 'spikeinterface.preprocessing.deepinterpolation.deepinterpolation.DeepInterpolatedRecording'>,
                        'filter': <class 'spikeinterface.preprocessing.filter.FilterRecording'>,
                        'highpass_filter': <class 'spikeinterface.preprocessing.filter.HighpassFilterRecording'>,
                        'normalize_by_quantile': <class 'spikeinterface.preprocessing.normalize_scale.NormalizeByQuantileRecording'>,
                        'notch_filter': <class 'spikeinterface.preprocessing.filter.NotchFilterRecording'>,
                        'phase_shift': <class 'spikeinterface.preprocessing.phase_shift.PhaseShiftRecording'>,
                        'rectify': <class 'spikeinterface.preprocessing.rectify.RectifyRecording'>,
                        'remove_artifacts': <class 'spikeinterface.preprocessing.remove_artifacts.RemoveArtifactsRecording'>,
                        'remove_bad_channels': <class 'spikeinterface.preprocessing.remove_bad_channels.RemoveBadChannelsRecording'>,
                        'resample': <class 'spikeinterface.preprocessing.resample.ResampleRecording'>,
                        'scale': <class 'spikeinterface.preprocessing.normalize_scale.ScaleRecording'>,
                        'whiten': <class 'spikeinterface.preprocessing.whiten.WhitenRecording'>,
                        'zero_channel_pad': <class 'spikeinterface.preprocessing.zero_channel_pad.ZeroChannelPaddedRecording
                    """
                    #******************** Current Implementation *************************
                    # if clustering_method.startswith('pykilosort'):
                    #     kilosort_triggering.run_pykilosort(
                    #         continuous_file=spikeglx_recording.root_dir / (
                    #                     spikeglx_recording.root_name + '.ap.bin'),
                    #         kilosort_output_directory=kilosort_dir,
                    #         channel_ind=params.pop('channel_ind'),
                    #         x_coords=params.pop('x_coords'),
                    #         y_coords=params.pop('y_coords'),
                    #         shank_ind=params.pop('shank_ind'),
                    #         connected=params.pop('connected'),
                    #         sample_rate=params.pop('sample_rate'),
                    #         params=params)
                    # elif clustering_method.startswith('kilosort'):
                    #     run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                    #         npx_input_dir=spikeglx_meta_filepath.parent,
                    #         ks_output_dir=kilosort_dir,
                    #         params=params,
                    #         KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                    #         run_CatGT=True)
                    #     run_kilosort.run_modules()

                    # ***************** New SpikeInterface Implementation ************************
                    
                    # Check through all types of spike sorters
                    if clustering_method.startswith('kilosort'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort",
                            recording = sglx_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort2'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort2",
                            recording = sglx_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort2-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort2_5'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort2_5",
                            recording = sglx_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort2_5-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort3'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort3",
                            recording = sglx_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort3-compiled-base:latest",
                            **params
                        )
                    else:
                        raise NotImplementedError(f'Automatic triggering of {clustering_method}'
                                          f' clustering analysis is not yet supported')
                    
                elif acq_software == 'Open Ephys':
                    # oe_probe = get_openephys_probe_data(key)

                    # inserted_probe_serial_number = (ProbeInsertion * probe.Probe
                    #                 & key).fetch1('probe')
                    # oe_file_path = find_full_path(get_ephys_root_data_dir(),
                    #                         get_session_directory(key))

                    # assert len(oe_probe.recording_info['recording_files']) == 1

                    # Clarify how OE views and loads multi probe data
                    
                    oe_filepath = get_openephys_filepath(key)

                    stream_name = os.path.split(oe_filepath)[1]
                    
                    oe_si_recording = se.OpenEphysBinaryRecordingExtractor(folder_path=oe_file_path, stream_name=stream_name) 

                    electrode_query = (probe.ProbeType.Electrode
                                    * probe.ElectrodeConfig.Electrode
                                    * EphysRecording & key)

                    xy_coords = [list(i) for i in zip(electrode_query.fetch('x_coord'),electrode_query.fetch('y_coord'))]

                    channel_details = get_recording_channels_details(key)
                        

                    probe = pi.Probe(ndim=2, si_units='um')
                    probe.set_contacts(positions=xy_coords, shapes='square', shape_params={'width': 5})
                    probe.create_auto_shape(probe_type='tip')


                    channel_indices = np.arange(channel_details['num_channels'])
                    probe.set_device_channel_indices(channel_indices)
                    oe_si_recording.set_probe(probe=probe)

                    # Kilosort Recording Object Preprocessing 
                    # Apply bandpass filter and rewrite recording object
                    # Store common mode referenced recording for construction of waveform extractor object
                    recording_cmr = oe_si_recording
                    oe_si_recording = sip.bandpass_filter(oe_si_recording, freq_min=300, freq_max=6000)
                    recording_cmr = sip.common_reference(oe_si_recording, reference="global", operator="median")


                    # run kilosort
                    #******************** Current Implementation *************************
                    # if clustering_method.startswith('pykilosort'):
                    #     kilosort_triggering.run_pykilosort(
                    #         continuous_file=pathlib.Path(oe_probe.recording_info['recording_files'][0]) / 'continuous.dat',
                    #         kilosort_output_directory=kilosort_dir,
                    #         channel_ind=params.pop('channel_ind'),
                    #         x_coords=params.pop('x_coords'),
                    #         y_coords=params.pop('y_coords'),
                    #         shank_ind=params.pop('shank_ind'),
                    #         connected=params.pop('connected'),
                    #         sample_rate=params.pop('sample_rate'),
                    #         params=params)
                    # else:
                    #     run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                    #         npx_input_dir=oe_probe.recording_info['recording_files'][0],
                    #         ks_output_dir=kilosort_dir,
                    #         params=params,
                    #         KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}')
                    #     run_kilosort.run_modules()

                    #******************** New SI Implementation *************************

                    if clustering_method.startswith('kilosort'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort",
                            recording = oe_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort2'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort2",
                            recording = oe_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort2-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort2_5'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort2_5",
                            recording = oe_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort2_5-compiled-base:latest",
                            **params
                        )
                    elif clustering_method.startswith('kilosort3'):
                        sorting_kilosort = si.run_sorter(
                            sorter_name = "kilosort3",
                            recording = oe_si_recording,
                            output_folder = kilosort_dir,
                            docker_image = "spikeinterface/kilosort3-compiled-base:latest",
                            **params
                        )
                    else:
                        raise NotImplementedError(f'Automatic triggering of {clustering_method}'
                                          f' clustering analysis is not yet supported')

            else:
                raise NotImplementedError(f'Automatic triggering of {clustering_method}'
                                          f' clustering analysis is not yet supported')

        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        creation_time, _, _ = kilosort.extract_clustering_info(kilosort_dir)
        self.insert1({**key, 'clustering_time': creation_time})

        if sorting_kilosort in locals():
            # Create Waveform Extraction Object
            we_kilosort = si.WaveformExtractor.create(recording_cmr, sorting_kilosort, "waveforms", remove_if_exists=True)
            we_kilosort.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
            we_kilosort.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
            unit_id0 = sorting_kilosort.unit_ids[0]
            waveforms = we_kilosort.get_waveforms(unit_id0)
            template = we_kilosort.get_template(unit_id0)
            snrs = si.compute_snrs(we_kilosort)
            

            # QC Metrics 
            si_violations_ratio, isi_violations_rate, isi_violations_count = si.compute_isi_violations(we_kilosort, isi_threshold_ms=1.5)
            metrics = si.compute_quality_metrics(we_kilosort, metric_names=["snr", "isi_violation", "amplitude_cutoff"])




@schema
class CuratedClustering(dj.Imported):
    definition = """
    # Clustering results of the spike sorting step.
    -> Clustering    
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
        spike_depths : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        output_dir = (ClusteringTask & key).fetch1('clustering_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)
        acq_software, sample_rate = (EphysRecording & key).fetch1(
            'acq_software', 'sampling_rate')

        sample_rate = kilosort_dataset.data['params'].get('sample_rate', sample_rate)

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
                                    / sample_rate)
                spike_count = len(unit_spike_times)

                units.append({
                    'unit': unit,
                    'cluster_quality_label': unit_lbl,
                    **channel2electrodes[unit_channel],
                    'spike_times': unit_spike_times,
                    'spike_count': spike_count,
                    'spike_sites': spike_sites[kilosort_dataset.data['spike_clusters'] == unit],
                    'spike_depths': spike_depths[kilosort_dataset.data['spike_clusters'] == unit]})

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
        output_dir = (ClusteringTask & key).fetch1('clustering_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)

        acq_software, probe_serial_number = (EphysRecording * ProbeInsertion & key).fetch1(
            'acq_software', 'probe')

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1('KEY')
        channel2electrodes = get_neuropixels_channel2electrode_map(recording_key, acq_software)

        # Get all units
        units = {u['unit']: u for u in (CuratedClustering.Unit & key).fetch(
            as_dict=True, order_by='unit')}

        if (kilosort_dir / 'mean_waveforms.npy').exists():
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
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)


# ---------------- HELPER FUNCTIONS ----------------

def get_spikeglx_meta_filepath(ephys_recording_key):
    # attempt to retrieve from EphysRecording.EphysFile
    spikeglx_meta_filepath = pathlib.Path(
        (
            EphysRecording.EphysFile
            & ephys_recording_key
            & 'file_path LIKE "%.ap.meta"'
        ).fetch1("file_path")
    )

    try:
        spikeglx_meta_filepath = find_full_path(get_ephys_root_data_dir(),
                                                spikeglx_meta_filepath)
    except FileNotFoundError:
        # if not found, search in session_dir again
        if not spikeglx_meta_filepath.exists():
            session_dir = find_full_path(get_ephys_root_data_dir(),
                                      get_session_directory(ephys_recording_key))
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


def get_openephys_filepath(ephys_recording_key):
     # attempt to retrieve from EphysRecording.EphysFile
    openephys_filepath = pathlib.Path(
        (
            EphysRecording.EphysFile
            & ephys_recording_key
            & 'file_path LIKE "%Neuropix-PXI%"'
        ).fetch1("file_path")
    )
    return openephys_filepath
    # try:
    #     openephys_meta_filepath = find_full_path(get_ephys_root_data_dir(),
    #                                             openephys_meta_filepath)
    # except FileNotFoundError:
    #     # if not found, search in session_dir again
    #     if not openephys_meta_filepath.exists():
    #         session_dir = find_full_path(get_ephys_root_data_dir(),
    #                                   get_session_directory(ephys_recording_key))
    #         inserted_probe_serial_number = (ProbeInsertion * probe.Probe
    #                                         & ephys_recording_key).fetch1('probe')

    #         openephys_meta_filepaths = [fp for fp in session_dir.rglob('*Neuropix-PXI-100')]
    #         for meta_filepath in openephys_meta_filepaths:
    #             openephys_meta = openephys.OpenEphys(meta_filepath)
    #             if str(openephys_meta.probe_SN) == inserted_probe_serial_number:
    #                 openephys_meta_filepath = meta_filepath
    #                 break
    #         else:
    #             raise FileNotFoundError(
    #                 'No OpenEphys data found for probe insertion: {}'.format(ephys_recording_key))
    # 
    # return open_meta_filepath


def get_openephys_probe_data(ephys_recording_key):
    inserted_probe_serial_number = (ProbeInsertion * probe.Probe
                                    & ephys_recording_key).fetch1('probe')
    session_dir = find_full_path(get_ephys_root_data_dir(),
                              get_session_directory(ephys_recording_key))
    loaded_oe = openephys.OpenEphys(session_dir)
    probe_data = loaded_oe.probes[inserted_probe_serial_number]

    # ASK thinh about 

    # explicitly garbage collect "loaded_oe"
    # as these may have large memory footprint and may not be cleared fast enough
    del loaded_oe
    gc.collect()

    return probe_data


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
        probe_dataset = get_openephys_probe_data(ephys_recording_key)

        electrode_query = (probe.ProbeType.Electrode
                           * probe.ElectrodeConfig.Electrode
                           * EphysRecording & ephys_recording_key)

        probe_electrodes = {key['electrode']: key
                            for key in electrode_query.fetch('KEY')}

        channel2electrode_map = {
            channel_idx: probe_electrodes[channel_idx]
            for channel_idx in probe_dataset.ap_meta['channels_indices']}

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


def get_recording_channels_details(ephys_recording_key):
    channels_details = {}

    acq_software, sample_rate = (EphysRecording & ephys_recording_key).fetch1('acq_software',
                                                              'sampling_rate')

    probe_type = (ProbeInsertion * probe.Probe & ephys_recording_key).fetch1('probe_type')
    channels_details['probe_type'] = {'neuropixels 1.0 - 3A': '3A',
                                      'neuropixels 1.0 - 3B': 'NP1',
                                      'neuropixels UHD': 'NP1100',
                                      'neuropixels 2.0 - SS': 'NP21',
                                      'neuropixels 2.0 - MS': 'NP24'}[probe_type]

    electrode_config_key = (probe.ElectrodeConfig * EphysRecording & ephys_recording_key).fetch1('KEY')
    channels_details['channel_ind'], channels_details['x_coords'], channels_details[
        'y_coords'], channels_details['shank_ind'] = (
            probe.ElectrodeConfig.Electrode * probe.ProbeType.Electrode
            & electrode_config_key).fetch('electrode', 'x_coord', 'y_coord', 'shank')
    channels_details['sample_rate'] = sample_rate
    channels_details['num_channels'] = len(channels_details['channel_ind'])

    if acq_software == 'SpikeGLX':
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
        channels_details['uVPerBit'] = spikeglx_recording.get_channel_bit_volts('ap')[0]
        channels_details['connected'] = np.array(
            [v for *_, v in spikeglx_recording.apmeta.shankmap['data']])
    elif acq_software == 'Open Ephys':
        oe_probe = get_openephys_probe_data(ephys_recording_key)
        channels_details['uVPerBit'] = oe_probe.ap_meta['channels_gains'][0]
        channels_details['connected'] = np.array([
            int(v == 1) for c, v in oe_probe.channels_connected.items()
            if c in channels_details['channel_ind']])

    return channels_details
