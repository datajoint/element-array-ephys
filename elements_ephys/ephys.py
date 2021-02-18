import datajoint as dj
import pathlib
import re
import numpy as np
import inspect
import uuid
import hashlib
import importlib

from .readers import neuropixels, kilosort
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
                + get_ephys_root_data_dir() -> str
                    Retrieve the root data directory - e.g. containing all subject/sessions data
                    :return: a string for full path to the root data directory
                + get_session_directory(session_key: dict) -> str
                    Retrieve the session directory containing the recorded Neuropixels data for a given Session
                    :param session_key: a dictionary of one Session `key`
                    :return: a string for full path to the session directory
                + get_kilosort_output_directory(clustering_task_key: dict) -> str
                    Retrieve the Kilosort output directory for a given ClusteringTask
                    :param clustering_task_key: a dictionary of one ClusteringTask `key`
                    :return: a string for full path to the resulting Kilosort output directory
                + get_paramset_idx(ephys_rec_key: dict) -> int
                    Retrieve attribute `paramset_idx` from the ClusteringParamSet record for the given EphysRecording.
                    :param ephys_rec_key: a dictionary of one EphysRecording `key`
                    :return: int specifying the `paramset_idx`
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    # activate
    probe.activate(probe_schema_name, create_schema=create_schema, create_tables=create_tables)
    schema.activate(ephys_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# -------------- Functions required by the elements-ephys  ---------------

def get_ephys_root_data_dir() -> str:
    """
    get_ephys_root_data_dir() -> str
        Retrieve the root data directory - e.g. containing all subject/sessions data
        :return: a string for full path to the root data directory
    """
    return _linking_module.get_ephys_root_data_dir()


def get_session_directory(session_key: dict) -> str:
    """
    get_session_directory(session_key: dict) -> str
        Retrieve the session directory containing the recorded Neuropixels data for a given Session
        :param session_key: a dictionary of one Session `key`
        :return: a string for full path to the session directory
    """
    return _linking_module.get_session_directory(session_key)


def get_kilosort_output_directory(clustering_task_key: dict) -> str:
    """
    get_kilosort_output_directory(clustering_task_key: dict) -> str
        Retrieve the Kilosort output directory for a given ClusteringTask
        :param clustering_task_key: a dictionary of one ClusteringTask `key`
        :return: a string for full path to the resulting Kilosort output directory
    """
    return _linking_module.get_kilosort_output_directory(clustering_task_key)


def get_paramset_idx(ephys_rec_key: dict) -> int:
    """
    get_paramset_idx(ephys_rec_key: dict) -> int
        Retrieve attribute `paramset_idx` from the ClusteringParamSet record for the given EphysRecording.
        :param ephys_rec_key: a dictionary of one EphysRecording `key`
        :return: int specifying the `paramset_idx`
    """
    return _linking_module.get_paramset_idx(ephys_rec_key)


# ----------------------------- Table declarations ----------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """  # Name of software used for recording of neuropixels probes - SpikeGLX or OpenEphys
    acq_software: varchar(24)    
    """
    contents = zip(['SpikeGLX', 'OpenEphys'])


@schema
class ProbeInsertion(dj.Manual):  # (acute)
    definition = """
    -> Session  
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """


@schema
class InsertionLocation(dj.Manual):

    definition = """
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
    -> ProbeInsertion      
    ---
    -> probe.ElectrodeConfig
    -> AcquisitionSoftware
    sampling_rate: float # (Hz) 
    """

    class EphysFile(dj.Part):
        definition = """
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    @property
    def key_source(self):
        return _linking_module.Session()

    def make(self, key):
        root_dir = pathlib.Path(get_ephys_root_data_dir())
        sess_dir = pathlib.Path(get_session_directory(key))

        meta_filepaths = list(sess_dir.rglob('*.ap.meta'))

        if len(meta_filepaths) == 0:
            raise FileNotFoundError(f'Ephys recording data not found! No ".ap.meta" files found in {sess_dir}')

        for meta_filepath in meta_filepaths:

            neuropixels_meta = neuropixels.NeuropixelsMeta(meta_filepath)
            insertion_key = (ProbeInsertion & key & {'probe': neuropixels_meta.probe_SN}).fetch1('KEY')

            if re.search('(1.0|2.0)', neuropixels_meta.probe_model):
                eg_members = []
                probe_type = {'probe_type': neuropixels_meta.probe_model}
                q_electrodes = probe.ProbeType.Electrode & probe_type
                for shank, shank_col, shank_row, is_used in neuropixels_meta.shankmap['data']:
                    electrode = (q_electrodes & {'shank': shank,
                                                 'shank_col': shank_col,
                                                 'shank_row': shank_row}).fetch1('KEY')
                    eg_members.append({**electrode, 'used_in_reference': is_used})
            else:
                raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(
                    neuropixels_meta.probe_model))

            # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
            ec_hash = dict_to_uuid({k['electrode']: k for k in eg_members})

            el_list = sorted([k['electrode'] for k in eg_members])
            el_jumps = [-1] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
            ec_name = '; '.join([f'{el_list[s + 1]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

            e_config = {'electrode_config_hash': ec_hash}

            # ---- make new ElectrodeConfig if needed ----
            if not probe.ElectrodeConfig & e_config:
                probe.ElectrodeConfig.insert1({**e_config, **probe_type, 'electrode_config_name': ec_name})
                probe.ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

            self.insert1({**insertion_key, **e_config,
                          'acq_software': 'SpikeGLX',
                          'sampling_rate': neuropixels_meta.meta['imSampRate']})
            self.EphysFile.insert1({**insertion_key, 'file_path': meta_filepath.relative_to(root_dir).as_posix()})


@schema
class LFP(dj.Imported):
    definition = """
    -> EphysRecording
    ---
    lfp_sampling_rate: float        # (Hz)
    lfp_time_stamps: longblob       # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob              # (mV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode  
        ---
        lfp: longblob               # (mV) recorded lfp at this electrode 
        """

    def make(self, key):
        root_dir = pathlib.Path(get_ephys_root_data_dir())

        npx_meta_fp = (EphysRecording.EphysFile & key & 'file_path LIKE "%.ap.meta"').fetch1('file_path')
        neuropixels_dir = (root_dir / npx_meta_fp).parent
        neuropixels_recording = neuropixels.Neuropixels(neuropixels_dir)

        lfp = neuropixels_recording.lfdata[:, :-1].T  # exclude the sync channel

        self.insert1(dict(key,
                          lfp_sampling_rate=neuropixels_recording.lfmeta['imSampRate'],
                          lfp_time_stamps=np.arange(lfp.shape[1]) / neuropixels_recording.lfmeta['imSampRate'],
                          lfp_mean=lfp.mean(axis=0)))
        '''
        Only store LFP for every 9th channel (defined in skip_chn_counts), counting in reverse
            Due to high channel density, close-by channels exhibit highly similar lfp
        '''
        q_electrodes = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode & key
        electrodes = []
        for recorded_site in np.arange(lfp.shape[0]):
            shank, shank_col, shank_row, _ = neuropixels_recording.npx_meta.shankmap['data'][recorded_site]
            electrodes.append((q_electrodes
                               & {'shank': shank,
                                  'shank_col': shank_col,
                                  'shank_row': shank_row}).fetch1('KEY'))

        chn_lfp = list(zip(electrodes, lfp))
        skip_chn_counts = 9
        self.Electrode().insert((
            {**key, **electrode, 'lfp': d}
            for electrode, d in chn_lfp[-1::-skip_chn_counts]), ignore_extra_fields=True)


# ------------ Clustering --------------

@schema
class ClusteringMethod(dj.Lookup):
    definition = """
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [('kilosort', 'kilosort clustering method'),
                ('kilosort2', 'kilosort2 clustering method')]


@schema
class ClusteringParamSet(dj.Lookup):
    definition = """
    paramset_idx:  smallint
    ---
    -> ClusteringMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, processing_method: str, paramset_idx: int, paramset_desc: str, params: dict):
        param_dict = {'clustering_method': processing_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(params)}
        q_param = cls & {'param_set_hash': param_dict['param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('paramset_idx')
            if pname == paramset_idx:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError('The specified param-set already exists - paramset_idx: {}'.format(pname))
        else:
            cls.insert1(param_dict)


@schema
class ClusteringTask(dj.Imported):
    definition = """
    -> EphysRecording
    ---
    -> ClusteringParamSet
    """

    def make(self, key):
        key['paramset_idx'] = get_paramset_idx(key)

        data_dir = pathlib.Path(get_kilosort_output_directory(key))
        if not data_dir.exists():
            # this is where we can trigger the clustering job
            print(f'Clustering results not yet found for {key} - Triggering kilosort not yet implemented!')

        self.insert1(key)


@schema
class ClusterQualityLabel(dj.Lookup):
    definition = """
    # Quality
    cluster_quality_label  :  varchar(100)
    ---
    cluster_quality_description :  varchar(4000)
    """
    contents = [
        ('good', 'single unit'),
        ('ok', 'probably a single unit, but could be contaminated'),
        ('mua', 'multi-unit activity'),
        ('noise', 'bad unit')
    ]


@schema
class Clustering(dj.Imported):
    definition = """
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results 
    quality_control: bool      # has this clustering result undergone quality control?
    manual_curation: bool      # has manual curation been performed on this clustering result?
    clustering_note='': varchar(2000)  
    """

    class Unit(dj.Part):
        definition = """   
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode on the probe that this unit has highest response amplitude
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording of this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        ks_dir = get_kilosort_output_directory(key)
        ks = kilosort.Kilosort(ks_dir)
        # ---------- Clustering ----------
        creation_time, is_curated, is_qc = kilosort.extract_clustering_info(ks_dir)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [i for i, u in enumerate(ks.data['cluster_ids']) if (ks.data['spike_clusters'] == u).any()]
        valid_units = ks.data['cluster_ids'][withspike_idx]
        valid_unit_labels = ks.data['cluster_groups'][withspike_idx]
        # -- Get channel and electrode-site mapping
        chn2electrodes = get_neuropixels_chn2electrode_map(key)

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spk_time_key = ('spike_times_sec_adj' if 'spike_times_sec_adj' in ks.data
                        else 'spike_times_sec' if 'spike_times_sec' in ks.data else 'spike_times')
        spike_times = ks.data[spk_time_key]
        ks.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = np.array([chn2electrodes[s]['electrode'] for s in ks.data['spike_sites']])
        spike_depths = ks.data['spike_depths']

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (ks.data['spike_clusters'] == unit).any():
                unit_channel, _ = ks.get_best_channel(unit)
                unit_spike_times = (spike_times[ks.data['spike_clusters'] == unit] / ks.data['params']['sample_rate'])
                spike_count = len(unit_spike_times)

                units.append({'unit': unit,
                              'cluster_quality_label': unit_lbl,
                              **chn2electrodes[unit_channel],
                              'spike_times': unit_spike_times,
                              'spike_count': spike_count,
                              'spike_sites': spike_sites[ks.data['spike_clusters'] == unit],
                              'spike_depths': spike_depths[ks.data['spike_clusters'] == unit]})

        self.insert1({**key, 'clustering_time': creation_time,
                      'quality_control': is_qc, 'manual_curation': is_curated})
        self.Unit.insert([{**key, **u} for u in units])


@schema
class Waveform(dj.Imported):
    definition = """
    -> Clustering.Unit
    ---
    peak_chn_waveform_mean: longblob  # mean over all spikes at the peak channel for this unit
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # mean over all spikes
        waveforms=null: longblob  # (spike x sample) waveform of each spike at each electrode
        """

    @property
    def key_source(self):
        return Clustering()

    def make(self, key):
        units = {u['unit']: u for u in (Clustering.Unit & key).fetch(as_dict=True, order_by='unit')}

        neuropixels_dir = get_neuropixels_data_directory(key)

        ks_dir = get_kilosort_output_directory(key)
        ks = kilosort.Kilosort(ks_dir)

        # -- Get channel and electrode-site mapping
        rec_key = (EphysRecording & key).fetch1('KEY')
        chn2electrodes = get_neuropixels_chn2electrode_map(rec_key)

        is_qc = (Clustering & key).fetch1('quality_control')

        unit_waveforms, unit_peak_waveforms = [], []
        if is_qc:
            unit_wfs = np.load(ks_dir / 'mean_waveforms.npy')  # unit x channel x sample
            for unit_no, unit_wf in zip(ks.data['cluster_ids'], unit_wfs):
                if unit_no in units:
                    for chn, chn_wf in zip(ks.data['channel_map'], unit_wf):
                        unit_waveforms.append({**units[unit_no], **chn2electrodes[chn], 'waveform_mean': chn_wf})
                        if chn2electrodes[chn]['electrode'] == units[unit_no]['electrode']:
                            unit_peak_waveforms.append({**units[unit_no], 'peak_chn_waveform_mean': chn_wf})
        else:
            neuropixels_recording = neuropixels.Neuropixels(neuropixels_dir)
            for unit_no, unit_dict in units.items():
                spks = (Clustering.Unit & unit_dict).fetch1('unit_spike_times')
                wfs = neuropixels_recording.extract_spike_waveforms(spks, ks.data['channel_map'])  # (sample x channel x spike)
                wfs = wfs.transpose((1, 2, 0))  # (channel x spike x sample)
                for chn, chn_wf in zip(ks.data['channel_map'], wfs):
                    unit_waveforms.append({**unit_dict, **chn2electrodes[chn],
                                           'waveform_mean': chn_wf.mean(axis=0),
                                           'waveforms': chn_wf})
                    if chn2electrodes[chn]['electrode'] == unit_dict['electrode']:
                        unit_peak_waveforms.append({**unit_dict, 'peak_chn_waveform_mean': chn_wf.mean(axis=0)})

        self.insert(unit_peak_waveforms, ignore_extra_fields=True)
        self.Electrode.insert(unit_waveforms, ignore_extra_fields=True)


# ----------- Quality Control ----------

@schema
class ClusterQualityMetrics(dj.Imported):
    definition = """
    -> Clustering.Unit
    ---
    amp: float
    snr: float
    isi_violation: float
    firing_rate: float

    presence_ratio: float  # Fraction of epoch in which spikes are present
    amplitude_cutoff: float  # Estimate of miss rate based on amplitude histogram
    isolation_distance=null: float  # Distance to nearest cluster in Mahalanobis space
    l_ratio=null: float  # 
    d_prime=null: float  # Classification accuracy based on LDA
    nn_hit_rate=null: float  # 
    nn_miss_rate=null: float
    silhouette_score=null: float  # Standard metric for cluster overlap
    max_drift=null: float  # Maximum change in spike depth throughout recording
    cumulative_drift=null: float  # Cumulative change in spike depth throughout recording 
    """

    @property
    def key_source(self):
        return Clustering

    def make(self, key):
        pass

# ---------------- HELPER FUNCTIONS ----------------


def get_neuropixels_chn2electrode_map(ephys_recording_key):
    neuropixels_dir = get_neuropixels_data_directory(ephys_recording_key)
    meta_filepath = next(pathlib.Path(neuropixels_dir).glob('*.ap.meta'))
    neuropixels_meta = neuropixels.NeuropixelsMeta(meta_filepath)
    e_config_key = (EphysRecording * probe.ElectrodeConfig & ephys_recording_key).fetch1('KEY')

    q_electrodes = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode & e_config_key
    chn2electrode_map = {}
    for recorded_site, (shank, shank_col, shank_row, _) in enumerate(neuropixels_meta.shankmap['data']):
        chn2electrode_map[recorded_site] = (q_electrodes
                                            & {'shank': shank,
                                               'shank_col': shank_col,
                                               'shank_row': shank_row}).fetch1('KEY')
    return chn2electrode_map


def dict_to_uuid(key):
    """
    Given a dictionary `key`, returns a hash string as UUID
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest())
