from open_ephys.analysis import Session as oeSession
import xmltodict
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
        self.root_dir = pathlib.Path(experiment_dir)

        settings_fp = self.root_dir / 'settings.xml'
        if not settings_fp.exists():
            raise FileNotFoundError(f'"settings.xml" not found for the OpenEphys session at: {self.root_dir}')
        with open(settings_fp) as f:
            self.settings = xmltodict.parse(f.read())['SETTINGS']

        oe_session = oeSession(self.root_dir.parent)  # this is on the Record Node level

        # extract the "recordings" for this session
        self.oe_recordings = [r for r in oe_session.recordings if pathlib.Path(r.directory).parent == self.root_dir]
        # extract probe data
        self.probes = self.load_probe_data()

    def load_probe_data(self):
        probes = []
        for processor in self.settings['SIGNALCHAIN']['PROCESSOR']:
            if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                probe = {'processor_id': int(processor['@NodeId']),
                         'ap_meta': None, 'lf_meta': None,
                         'ap_data': [], 'lf_data': [],
                         'recording_duration': 0}
                if processor['@pluginName'] == 'Neuropix-PXI':
                    probe['probe_model'] = 'neuropixels 1.0 - 3B'
                elif processor['@pluginName'] == 'Neuropix-3a':
                    probe['probe_model'] = 'neuropixels 1.0 - 3A'
                # TODO: how to determine npx 2.0 SS and MS?

                probe['probe_info'] = processor['EDITOR']['PROBE']
                probe['probe_SN'] = probe['probe_info']['@probe_serial_number']

                t_start, t_end = 0, 0
                for rec in self.oe_recordings:
                    for cont_info, cont in zip(rec.info['continuous'], rec.continuous):
                        if cont.metadata['processor_id'] != probe['processor_id']:
                            continue

                        if cont.metadata['subprocessor_id'] == 0:  # ap data
                            assert cont_info['sample_rate'] == cont.metadata['sample_rate'] == 30000
                            prefix = 'ap_'
                            #
                            t_start = (cont.timestamps[0] / cont_info['sample_rate']
                                       if cont.timestamps[0] / cont_info['sample_rate'] < t_start
                                       else t_start)
                            t_end = (cont.timestamps[-1] / cont_info['sample_rate']
                                     if cont.timestamps[-1] / cont_info['sample_rate'] > t_end
                                     else t_end)
                        elif cont.metadata['subprocessor_id'] == 1:  # lfp data
                            assert cont_info['sample_rate'] == cont.metadata['sample_rate'] == 2500
                            prefix = 'lfp_'

                        probe[prefix + 'meta'] = cont_info
                        probe[prefix + 'data'].append(cont)

                probe['recording_duration'] = t_start - t_end
                probes.append(probe)

        return probes
