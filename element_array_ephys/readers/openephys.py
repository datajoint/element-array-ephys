import pathlib
import pyopenephys
import numpy as np


"""
The Open Ephys Record Node saves Neuropixels data in binary format according to the following the directory structure:
(https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html)

Record Node 102
-- experiment1 (equivalent to a Session)
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
        self.sess_dir = pathlib.Path(experiment_dir)

        oe_file = pyopenephys.File(self.sess_dir.parent)  # this is on the Record Node level

        # extract the "recordings" for this session
        self.experiment = next(experiment for experiment in oe_file.experiments
                               if pathlib.Path(experiment.absolute_foldername) == self.sess_dir)

        self.recording_time = self.experiment.datetime

        # extract probe data
        self.probes = self.load_probe_data()

    def load_probe_data(self):
        """
        Loop through all Open Ephys "processors", identify the processor for
         the Neuropixels probe(s), extract probe info
            Loop through all recordings, associate recordings to
            the matching probes, extract recording info

        Yielding multiple "Probe" objects, each containing meta information
         and timeseries data associated with each probe
        """

        probes = {}
        for processor in self.experiment.settings['SIGNALCHAIN']['PROCESSOR']:
            if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                if processor['@pluginName'] == 'Neuropix-3a':
                    oe_probe = Probe(processor)
                    probes[oe_probe.probeSN] = oe_probe
                else:
                    for probe_index in range(len(processor['EDITOR']['NP_PROBE'])):
                        oe_probe = Probe(processor, probe_index)
                        probes[oe_probe.probe_SN] = oe_probe
                        
        for probe_index, probe_SN in enumerate(probes.keys()):
            
            oe_probe = probes[probe_SN]
                    
            for rec in self.experiment.recordings:
                for cont_info, analog_signal in zip(rec._oebin['continuous'],
                                                    rec.analog_signals):
                    if cont_info['source_processor_id'] != oe_probe.processor_id:
                        continue

                    if cont_info['source_processor_sub_idx'] == probe_index * 2:  # ap data
                        assert cont_info['sample_rate'] == analog_signal.sample_rate == 30000
                        cont_type = 'ap'

                        oe_probe.recording_info['recording_count'] += 1
                        oe_probe.recording_info['recording_datetimes'].append(
                            rec.datetime)
                        oe_probe.recording_info['recording_durations'].append(
                            float(rec.duration))
                        oe_probe.recording_info['recording_files'].append(
                            rec.absolute_foldername / 'continuous' / cont_info['folder_name'])

                    elif cont_info['source_processor_sub_idx'] == probe_index * 2 + 1:  # lfp data
                        assert cont_info['sample_rate'] == analog_signal.sample_rate == 2500
                        cont_type = 'lfp'

                    if getattr(oe_probe, cont_type + '_meta') is None:
                        cont_info['channels_ids'] = analog_signal.channel_ids
                        cont_info['channels_names'] = analog_signal.channel_names
                        cont_info['channels_gains'] = analog_signal.gains
                        setattr(oe_probe, cont_type + '_meta', cont_info)

                    oe_probe.__dict__[f'{cont_type}_analog_signals'].append(analog_signal)

        return probes


class Probe:

    def __init__(self, processor, probe_index=0):
        self.processor_id = int(processor['@NodeId'])
        
        if processor['@pluginName'] == 'Neuropix-3a':
            
            self.probe_info = processor['EDITOR']['PROBE']
            self.probe_SN = self.probe_info['@probe_serial_number']
            self.probe_model = "Neuropixels 3A"
        else:
            self.probe_info = processor['EDITOR']['NP_PROBE'][probe_index]
            self.probe_SN = self.probe_info['@probe_serial_number']
            self.probe_model = self.probe_info['@probe_name']

        self.ap_meta = None
        self.lfp_meta = None

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
