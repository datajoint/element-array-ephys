import pathlib
import pyopenephys
import numpy as np


"""
OpenEphys plug-in for Neuropixels recording saves data in binary format the following the directory structure:
(https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html)

Record Node 102
-- experiment1 (equivalent to a Session)
   -- recording1
   -- recording2
      -- continuous
         -- Neuropix-PXI-100.0 (probe0 ap)
         -- Neuropix-PXI-100.1 (probe0 lf)
         -- Neuropix-PXI-101.0 (probe1 ap)
         -- Neuropix-PXI-101.1 (probe1 lf)
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
        self.experiment = [experiment for experiment in oe_file.experiments
                           if pathlib.Path(experiment.absolute_foldername) == self.sess_dir][0]

        self.recording_time = self.experiment.datetime

        # extract probe data
        self.probes = self.load_probe_data()

    def load_probe_data(self):
        probes = {}
        for processor in self.experiment.settings['SIGNALCHAIN']['PROCESSOR']:
            if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                oe_probe = Probe(processor)

                for rec in self.experiment.recordings:
                    for cont_info, analog_signal in zip(rec._oebin['continuous'], rec.analog_signals):
                        if cont_info['source_processor_id'] != oe_probe.processor_id:
                            continue

                        if cont_info['source_processor_sub_idx'] == 0:  # ap data
                            assert cont_info['sample_rate'] == analog_signal.sample_rate == 30000
                            cont_type = 'ap'

                            oe_probe.recording_info['recording_count'] += 1
                            oe_probe.recording_info['recording_datetimes'].append(rec.datetime)
                            oe_probe.recording_info['recording_durations'].append(float(rec.duration))
                            oe_probe.recording_info['recording_files'].append(
                                rec.absolute_foldername / cont_info['folder_name'])

                        elif cont_info['source_processor_sub_idx'] == 1:  # lfp data
                            assert cont_info['sample_rate'] == analog_signal.sample_rate == 2500
                            cont_type = 'lfp'

                        if getattr(oe_probe, cont_type + '_meta') is None:
                            cont_info['channels_ids'] = analog_signal.channel_ids
                            cont_info['channels_names'] = analog_signal.channel_names
                            cont_info['channels_gains'] = analog_signal.gains
                            setattr(oe_probe, cont_type + '_meta', cont_info)

                        oe_probe.__dict__[f'{cont_type}_analog_signals'].append(analog_signal)

                probes[oe_probe.probe_SN] = oe_probe

        return probes


class Probe:

    def __init__(self, processor):
        self.processor_id = int(processor['@NodeId'])
        self.probe_info = processor['EDITOR']['PROBE']
        self.probe_SN = self.probe_info['@probe_serial_number']

        # Determine probe-model (TODO: how to determine npx 2.0 SS and MS?)
        if processor['@pluginName'] == 'Neuropix-PXI':
            self.probe_model = 'neuropixels 1.0 - 3B'
        elif processor['@pluginName'] == 'Neuropix-3a':
            self.probe_model = 'neuropixels 1.0 - 3A'

        self.ap_meta = None
        self.lfp_meta = None

        self.ap_analog_signals = []
        self.lfp_analog_signals = []

        self.recording_info = {'recording_count': 0,
                               'recording_datetimes': [],
                               'recording_durations': [],
                               'recording_files': []}

        self._ap_data = None
        self._ap_timestamps = None
        self._lfp_data = None
        self._lfp_timestamps = None

    @property
    def ap_data(self):
        if self._ap_data is None:
            self._ap_data = np.hstack([s.signal for s in self.ap_analog_signals])
        return self._ap_data

    @property
    def ap_timestamps(self):
        if self._ap_timestamps is None:
            self._ap_timestamps = np.hstack([s.times for s in self.ap_analog_signals])
        return self._ap_timestamps

    @property
    def lfp_data(self):
        if self._lfp_data is None:
            self._lfp_data = np.hstack([s.signal for s in self.lfp_analog_signals])
        return self._lfp_data

    @property
    def lfp_timestamps(self):
        if self._lfp_timestamps is None:
            self._lfp_timestamps = np.hstack([s.times for s in self.lfp_analog_signals])
        return self._lfp_timestamps