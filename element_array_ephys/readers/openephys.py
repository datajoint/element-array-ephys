import pathlib
import pyopenephys
import numpy as np
import re
import datetime


"""
The Open Ephys Record Node saves Neuropixels data in binary format according to the following the directory structure:
(https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html)

Record Node 102
-- experiment1 (equivalent to one experimental session - multi probes, multi recordings per probe)
   -- recording1
   -- recording2
      -- continuous
         -- Neuropix-PXI-100.0 (probe0 ap)
         -- Neuropix-PXI-100.1 (probe0 lf)
         -- Neuropix-PXI-100.2 (probe1 ap)
         -- Neuropix-PXI-100.3 (probe1 lf)
         ...
      -- events
      -- spikes
      -- structure.oebin
-- experiment 2
   ...
-- settings.xml
-- settings2.xml
...
"""


class OpenEphys:

    def __init__(self, experiment_dir):
        self.session_dir = pathlib.Path(experiment_dir)

        openephys_file = pyopenephys.File(self.session_dir.parent)  # this is on the Record Node level

        # extract the "recordings" for this session
        self.experiment = next(experiment for experiment in openephys_file.experiments
                               if pathlib.Path(experiment.absolute_foldername) == self.session_dir)

        # extract probe data
        self.probes = self.load_probe_data()

        #
        self._recording_time = None

    @property
    def recording_time(self):
        if self._recording_time is None:
            recording_datetimes = []
            for probe in self.probes.values():
                recording_datetimes.extend(probe.recording_info['recording_datetimes'])
            self._recording_time = sorted(recording_datetimes)[0]
        return self._recording_time

    def load_probe_data(self):
        """
        Loop through all Open Ephys "signalchains/processors", identify the processor for
         the Neuropixels probe(s), extract probe info
            Loop through all recordings, associate recordings to
            the matching probes, extract recording info

        Yielding multiple "Probe" objects, each containing meta information
         and timeseries data associated with each probe
        """

        probes = {}
        sigchain_iter = (self.experiment.settings['SIGNALCHAIN']
                         if isinstance(self.experiment.settings['SIGNALCHAIN'], list)
                         else [self.experiment.settings['SIGNALCHAIN']])
        for sigchain in sigchain_iter:
            processor_iter = (sigchain['PROCESSOR']
                              if isinstance(sigchain['PROCESSOR'], list)
                              else [sigchain['PROCESSOR']])
            for processor in processor_iter:
                if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                    if (processor['@pluginName'] == 'Neuropix-3a'
                            or 'NP_PROBE' not in processor['EDITOR']):
                        for probe_index in range(len(processor['EDITOR']['PROBE'])):
                            probe = Probe(processor, probe_index)
                            probes[probe.probe_SN] = probe
                    else:
                        for probe_index in range(len(processor['EDITOR']['NP_PROBE'])):
                            probe = Probe(processor, probe_index)
                            probes[probe.probe_SN] = probe
                        
        for probe_index, probe_SN in enumerate(probes):
            
            probe = probes[probe_SN]
                    
            for rec in self.experiment.recordings:
                for continuous_info, analog_signal in zip(rec._oebin['continuous'],
                                                          rec.analog_signals):
                    if continuous_info['source_processor_id'] != probe.processor_id:
                        continue

                    if continuous_info['source_processor_sub_idx'] == probe_index * 2:  # ap data
                        assert continuous_info['sample_rate'] == analog_signal.sample_rate == 30000
                        continuous_type = 'ap'

                        probe.recording_info['recording_count'] += 1
                        probe.recording_info['recording_datetimes'].append(
                            rec.datetime + datetime.timedelta(seconds=float(rec.start_time)))
                        probe.recording_info['recording_durations'].append(
                            float(rec.duration))
                        probe.recording_info['recording_files'].append(
                            rec.absolute_foldername / 'continuous' / continuous_info['folder_name'])

                    elif continuous_info['source_processor_sub_idx'] == probe_index * 2 + 1:  # lfp data
                        assert continuous_info['sample_rate'] == analog_signal.sample_rate == 2500
                        continuous_type = 'lfp'

                    meta = getattr(probe, continuous_type + '_meta')
                    if not meta:
                        # channel indices - 0-based indexing
                        channels_indices = [int(re.search(r'\d+$', chn_name).group()) - 1
                                            for chn_name in analog_signal.channel_names]

                        meta.update(**continuous_info,
                                    channels_indices=channels_indices,
                                    channels_ids=analog_signal.channel_ids,
                                    channels_names=analog_signal.channel_names,
                                    channels_gains=analog_signal.gains)

                    signal = getattr(probe, continuous_type + '_analog_signals')
                    signal.append(analog_signal)

        return probes


class Probe:

    def __init__(self, processor, probe_index=0):
        self.processor_id = int(processor['@NodeId'])
        
        if processor['@pluginName'] == 'Neuropix-3a' or 'NP_PROBE' not in processor['EDITOR']:
            self.probe_info = processor['EDITOR']['PROBE'][probe_index]
            self.probe_SN = self.probe_info['@probe_serial_number']
            self.probe_model = {
                "Neuropix-PXI": "neuropixels 1.0 - 3B",
                "Neuropix-3a": "neuropixels 1.0 - 3A"}[processor['@pluginName']]
            self._channels_connected = {int(re.search(r'\d+$', k).group()): int(v)
                                        for k, v in self.probe_info.pop('CHANNELSTATUS').items()}
        else:
            self.probe_info = processor['EDITOR']['NP_PROBE'][probe_index]
            self.probe_SN = self.probe_info['@probe_serial_number']
            self.probe_model = {
                "Neuropixels 1.0": "neuropixels 1.0 - 3B",
                "Neuropixels Ultra": "neuropixels UHD",
                "Neuropixels 21": "neuropixels 2.0 - SS",
                "Neuropixels 24": "neuropixels 2.0 - MS"}[self.probe_info['@probe_name']]
            self._channels_connected = {int(re.search(r'\d+$', k).group()): 1
                                        for k in self.probe_info.pop('CHANNELS')}

        self.ap_meta = {}
        self.lfp_meta = {}

        self.ap_analog_signals = []
        self.lfp_analog_signals = []

        self.recording_info = {'recording_count': 0,
                               'recording_datetimes': [],
                               'recording_durations': [],
                               'recording_files': []}

        self._ap_timeseries = None
        self._ap_timestamps = None
        self._lfp_timeseries = None
        self._lfp_timestamps = None

    @property
    def channels_connected(self):
        return {chn_idx: self._channels_connected.get(chn_idx, 0)
                for chn_idx in self.ap_meta['channels_indices']}

    @property
    def ap_timeseries(self):
        """
        AP data concatenated across recordings. Shape: (sample x channel)
        Data are stored as int16 - to convert to microvolts,
         multiply with self.ap_meta['channels_gains']
        """
        if self._ap_timeseries is None:
            self._ap_timeseries = np.hstack([s.signal for s in self.ap_analog_signals]).T
        return self._ap_timeseries

    @property
    def ap_timestamps(self):
        if self._ap_timestamps is None:
            self._ap_timestamps = np.hstack([s.times for s in self.ap_analog_signals])
        return self._ap_timestamps

    @property
    def lfp_timeseries(self):
        """
        LFP data concatenated across recordings. Shape: (sample x channel)
        Data are stored as int16 - to convert to microvolts,
         multiply with self.lfp_meta['channels_gains']
        """
        if self._lfp_timeseries is None:
            self._lfp_timeseries = np.hstack([s.signal for s in self.lfp_analog_signals]).T
        return self._lfp_timeseries

    @property
    def lfp_timestamps(self):
        if self._lfp_timestamps is None:
            self._lfp_timestamps = np.hstack([s.times for s in self.lfp_analog_signals])
        return self._lfp_timestamps

    def extract_spike_waveforms(self, spikes, channel_ind, n_wf=500, wf_win=(-32, 32)):
        """
        :param spikes: spike times (in second) to extract waveforms
        :param channel_ind: channel indices (of meta['channels_ids']) to extract waveforms
        :param n_wf: number of spikes per unit to extract the waveforms
        :param wf_win: number of sample pre and post a spike
        :return: waveforms (sample x channel x spike)
        """
        channel_bit_volts = np.array(self.ap_meta['channels_gains'])[channel_ind]

        # ignore spikes at the beginning or end of raw data
        spikes = spikes[np.logical_and(spikes > (-wf_win[0] / self.ap_meta['sample_rate']),
                                       spikes < (self.ap_timestamps.max() - wf_win[-1]
                                                 / self.ap_meta['sample_rate']))]
        # select a randomized set of "n_wf" spikes
        np.random.shuffle(spikes)
        spikes = spikes[:n_wf]
        # extract waveforms
        if len(spikes) > 0:
            spike_indices = np.searchsorted(self.ap_timestamps, spikes, side="left")
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack([
                self.ap_timeseries[int(spk + wf_win[0]):int(spk + wf_win[-1]), channel_ind]
                * channel_bit_volts
                for spk in spike_indices])
            return spike_wfs
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            return np.full((len(range(*wf_win)), len(channel_ind), 1), np.nan)
