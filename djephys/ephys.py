import pathlib
import re
import numpy as np
import datajoint as dj
import uuid

from . import utils
from .probe import schema, Probe, ProbeType, ElectrodeConfig
from ephys_loaders import neuropixels, kilosort

from djutils.templates import required

# ===================================== Probe Insertion =====================================


@schema
class ProbeInsertion(dj.Manual):  # (acute)

    _Session = ...

    definition = """
    -> self._Session  
    insertion_number: tinyint unsigned
    ---
    -> Probe
    """


# ===================================== Insertion Location =====================================


@schema
class InsertionLocation(dj.Manual):

    _SkullReference = ...

    definition = """
    -> ProbeInsertion
    ---
    -> self._SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


# ===================================== Ephys Recording =====================================
# The abstract function _get_npx_data_dir() should expect one argument in the form of a
# dictionary with the keys from user-defined Subject and Session, as well as
# "insertion_number" (as int) based on the "ProbeInsertion" table definition in this djephys


@schema
class EphysRecording(dj.Imported):
    definition = """
    -> ProbeInsertion      
    ---
    -> ElectrodeConfig
    sampling_rate: float # (Hz) 
    """

    @staticmethod
    @required
    def _get_npx_data_dir():
        return None

    def make(self, key):
        npx_dir = EphysRecording._get_npx_data_dir(key)

        meta_filepath = next(pathlib.Path(npx_dir).glob('*.ap.meta'))

        npx_meta = neuropixels.NeuropixelsMeta(meta_filepath)

        if re.search('(1.0|2.0)', npx_meta.probe_model):
            eg_members = []
            probe_type = {'probe_type': npx_meta.probe_model}
            q_electrodes = ProbeType.Electrode & probe_type
            for shank, shank_col, shank_row, is_used in npx_meta.shankmap['data']:
                electrode = (q_electrodes & {'shank': shank,
                                             'shank_col': shank_col,
                                             'shank_row': shank_row}).fetch1('KEY')
                eg_members.append({**electrode, 'used_in_reference': is_used})
        else:
            raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(
                npx_meta.probe_model))

        # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
        ec_hash = uuid.UUID(utils.dict_to_hash({k['electrode']: k for k in eg_members}))

        el_list = sorted([k['electrode'] for k in eg_members])
        el_jumps = [-1] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
        ec_name = '; '.join([f'{el_list[s + 1]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

        e_config = {'electrode_config_hash': ec_hash}

        # ---- make new ElectrodeConfig if needed ----
        if not (ElectrodeConfig & e_config):
            ElectrodeConfig.insert1({**e_config, **probe_type, 'electrode_config_name': ec_name})
            ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

        self.insert1({**key, **e_config, 'sampling_rate': npx_meta.meta['imSampRate']})


# ===========================================================================================
# ================================= NON-CONFIGURABLE COMPONENTS =============================
# ===========================================================================================


# ===================================== Ephys LFP =====================================

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
        -> ElectrodeConfig.Electrode  
        ---
        lfp: longblob               # (mV) recorded lfp at this electrode 
        """

    def make(self, key):
        npx_dir = EphysRecording._get_npx_data_dir(key)
        npx_recording = neuropixels.Neuropixels(npx_dir)

        lfp = npx_recording.lfdata[:, :-1].T  # exclude the sync channel

        self.insert1(dict(key,
                          lfp_sampling_rate=npx_recording.lfmeta['imSampRate'],
                          lfp_time_stamps=np.arange(lfp.shape[1]) / npx_recording.lfmeta['imSampRate'],
                          lfp_mean=lfp.mean(axis=0)))
        '''
        Only store LFP for every 9th channel (defined in skip_chn_counts), counting in reverse
            Due to high channel density, close-by channels exhibit highly similar lfp
        '''
        q_electrodes = ProbeType.Electrode * ElectrodeConfig.Electrode & key
        electrodes = []
        for recorded_site in np.arange(lfp.shape[0]):
            shank, shank_col, shank_row, _ = npx_recording.npx_meta.shankmap['data'][recorded_site]
            electrodes.append((q_electrodes
                               & {'shank': shank,
                                  'shank_col': shank_col,
                                  'shank_row': shank_row}).fetch1('KEY'))

        chn_lfp = list(zip(electrodes, lfp))
        skip_chn_counts = 9
        self.Electrode.insert(({**key, **electrode, 'lfp': d}
                               for electrode, d in chn_lfp[-1::-skip_chn_counts]), ignore_extra_fields=True)


# ===================================== Clustering =====================================


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
                      'param_set_hash': uuid.UUID(utils.dict_to_hash(params))}
        q_param = cls & {'param_set_hash': param_dict['param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('param_set_name')
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError('The specified param-set already exists - name: {}'.format(pname))
        else:
            cls.insert1(param_dict)


@schema
class ClusteringTask(dj.Imported):
    definition = """
    -> EphysRecording
    ---
    -> ClusteringParamSet
    """

    @staticmethod
    @required
    def _get_paramset_idx(ephys_rec_key: dict) -> int:
        """
        Retrieve the 'paramset_idx' (for ClusteringParamSet) to be used for this EphysRecording
        :param ephys_rec_key: a dictionary of one EphysRecording
        :return: int specifying the 'paramset_idx'
        """
        return None

    @staticmethod
    @required
    def _get_ks_data_dir(clustering_task_key: dict) -> str:
        """
        Retrieve the Kilosort output directory for a given ClusteringTask
        :param clustering_task_key: a dictionary of one EphysRecording
        :return: a string for full path to the resulting Kilosort output directory
        """
        return None

    def make(self, key):
        key['paramset_idx'] = ClusteringTask._get_paramset_idx(key)
        self.insert1(key)

        data_dir = pathlib.Path(ClusteringTask._get_ks_data_dir(key))
        if not data_dir.exists():
            # this is where we can trigger the clustering job
            print(f'Clustering results not yet found for {key} - Triggering kilosort not yet implemented!')


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
        -> ElectrodeConfig.Electrode  # electrode on the probe that this unit has highest response amplitude
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording of this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        ks_dir = ClusteringTask._get_ks_data_dir(key)
        ks = kilosort.Kilosort(ks_dir)
        # ---------- Clustering ----------
        creation_time, is_curated, is_qc = kilosort.extract_clustering_info(ks_dir)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [i for i, u in enumerate(ks.data['cluster_ids']) if (ks.data['spike_clusters'] == u).any()]
        valid_units = ks.data['cluster_ids'][withspike_idx]
        valid_unit_labels = ks.data['cluster_groups'][withspike_idx]
        # -- Get channel and electrode-site mapping
        chn2electrodes = get_npx_chn2electrode_map(key)

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
        -> ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # mean over all spikes
        waveforms=null: longblob  # (spike x sample) waveform of each spike at each electrode
        """

    @property
    def key_source(self):
        return Clustering

    def make(self, key):
        units = {u['unit']: u for u in (Clustering.Unit & key).fetch(as_dict=True, order_by='unit')}

        npx_dir = EphysRecording._get_npx_data_dir(key)
        meta_filepath = next(pathlib.Path(npx_dir).glob('*.ap.meta'))
        npx_meta = neuropixels.NeuropixelsMeta(meta_filepath)

        ks_dir = ClusteringTask._get_ks_data_dir(key)
        ks = kilosort.Kilosort(ks_dir)

        # -- Get channel and electrode-site mapping
        rec_key = (EphysRecording & key).fetch1('KEY')
        chn2electrodes = get_npx_chn2electrode_map(rec_key)

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
            npx_recording = neuropixels.Neuropixels(npx_dir)
            for unit_no, unit_dict in units.items():
                spks = (Clustering.Unit & unit_dict).fetch1('unit_spike_times')
                wfs = npx_recording.extract_spike_waveforms(spks, ks.data['channel_map'])  # (sample x channel x spike)
                wfs = wfs.transpose((1, 2, 0))  # (channel x spike x sample)
                for chn, chn_wf in zip(ks.data['channel_map'], wfs):
                    unit_waveforms.append({**unit_dict, **chn2electrodes[chn],
                                           'waveform_mean': chn_wf.mean(axis=0),
                                           'waveforms': chn_wf})
                    if chn2electrodes[chn]['electrode'] == unit_dict['electrode']:
                        unit_peak_waveforms.append({**unit_dict, 'peak_chn_waveform_mean': chn_wf.mean(axis=0)})

        self.insert(unit_peak_waveforms, ignore_extra_fields=True)
        self.Electrode.insert(unit_waveforms, ignore_extra_fields=True)


# ===================================== Quality Control [WIP] =====================================

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

# ========================== HELPER FUNCTIONS =======================


def get_npx_chn2electrode_map(ephys_recording_key):
    npx_dir = EphysRecording._get_npx_data_dir(ephys_recording_key)
    meta_filepath = next(pathlib.Path(npx_dir).glob('*.ap.meta'))
    npx_meta = neuropixels.NeuropixelsMeta(meta_filepath)
    e_config_key = (EphysRecording * ElectrodeConfig & ephys_recording_key).fetch1('KEY')

    q_electrodes = ProbeType.Electrode * ElectrodeConfig.Electrode & e_config_key
    chn2electrode_map = {}
    for recorded_site, (shank, shank_col, shank_row, _) in enumerate(npx_meta.shankmap['data']):
        chn2electrode_map[recorded_site] = (q_electrodes
                                            & {'shank': shank,
                                               'shank_col': shank_col,
                                               'shank_row': shank_row}).fetch1('KEY')
    return chn2electrode_map
