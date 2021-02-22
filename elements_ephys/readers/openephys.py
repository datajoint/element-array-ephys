import pathlib
import pyopenephys


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
        probes = []
        for processor in self.experiment.settings['SIGNALCHAIN']['PROCESSOR']:
            if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                probe = {'processor_id': int(processor['@NodeId']),
                         'ap_meta': None, 'lfp_meta': None,
                         'ap_data': [], 'lfp_data': []}
                if processor['@pluginName'] == 'Neuropix-PXI':
                    probe['probe_model'] = 'neuropixels 1.0 - 3B'
                elif processor['@pluginName'] == 'Neuropix-3a':
                    probe['probe_model'] = 'neuropixels 1.0 - 3A'
                # TODO: how to determine npx 2.0 SS and MS?

                probe['probe_info'] = processor['EDITOR']['PROBE']
                probe['probe_SN'] = probe['probe_info']['@probe_serial_number']
                probe['recording_info'] = {'recording_count': 0,
                                           'recording_datetimes': [],
                                           'recording_durations': [],
                                           'recording_filepaths': []}

                for rec in self.experiment.recordings:
                    for cont_info, analog_signal in zip(rec._oebin['continuous'], rec.analog_signals):
                        if cont_info['source_processor_id'] != probe['processor_id']:
                            continue

                        if cont_info['source_processor_sub_idx'] == 0:  # ap data
                            assert cont_info['sample_rate'] == analog_signal.sample_rate == 30000
                            prefix = 'ap_'

                            probe['recording_info']['recording_count'] += 1
                            probe['recording_info']['recording_datetimes'].append(rec.datetime)
                            probe['recording_info']['recording_durations'].append(float(rec.duration))
                            probe['recording_info']['recording_files'].append(
                                rec.absolute_foldername / cont_info['folder_name'])

                        elif cont_info['source_processor_sub_idx'] == 1:  # lfp data
                            assert cont_info['sample_rate'] == analog_signal.sample_rate == 2500
                            prefix = 'lfp_'

                        if probe[prefix + 'meta'] is None:
                            cont_info['channels_ids'] = analog_signal.channel_ids
                            cont_info['channels_names'] = analog_signal.channel_names
                            cont_info['channels_gains'] = analog_signal.gains
                            probe[prefix + 'meta'] = cont_info

                        probe[prefix + 'data'].append(analog_signal)

                probes.append(probe)

        return probes


