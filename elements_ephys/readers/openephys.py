from open_ephys.analysis import Session as oeSession
import xmltodict
import pathlib
import itertools


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
"""


class OpenEphys:

    def __init__(self, experiment_dir):
        self.root_dir = pathlib.Path(experiment_dir)

        settings_fp = self.root_dir / 'settings.xml'
        if not settings_fp.exists():
            raise FileNotFoundError(f'"settings.xml" not found for the OpenEphys session at: {self.root_dir}')
        with open(settings_fp) as f:
            self.settings = xmltodict.parse(f.read())

        oe_session = oeSession(self.root_dir.parent)  # this is on the Record Node level

        # extract the "recordings" for this session
        recordings = [r for r in oe_session.recordings if pathlib.Path(r.directory).parent == self.root_dir]

        probes = []
        for processor in self.settings['SETTINGS']['SIGNALCHAIN']['PROCESSOR']:
            if processor['@pluginName'] in ('Neuropix-PXI', 'Neuropix-3a'):
                probe = {'processor_id': int(processor['@NodeId'])}
                if processor['@pluginName'] == 'Neuropix-PXI':
                    probe['probe_model'] = 'neuropixels 1.0 - 3B'
                elif processor['@pluginName'] == 'Neuropix-3a':
                    probe['probe_model'] = 'neuropixels 1.0 - 3A'
                # TODO: how to determine npx 2.0 SS and MS?

                probe['probe_info'] = processor['EDITOR']['PROBE']
                probe['probe_SN'] = probe['probe_info']['@probe_serial_number']

                for r in recordings:
                    for c_info, c in zip(r.info['continuous'], r.continuous):
                        if c.metadata['processor_id'] != probe['processor_id']:
                            continue

                        if c.metadata['subprocessor_id'] == 0:  # ap data
                            assert c_info['sample_rate'] == c.metadata['sample_rate'] == 30000
                            probe['ap_meta'] = c_info
                            if 'ap_data' in probe:
                                probe['ap_data'].append(c)
                            else:
                                probe['ap_data'] = [c]
                        elif c.metadata['subprocessor_id'] == 1:  # lf data
                            assert c_info['sample_rate'] == c.metadata['sample_rate'] == 2500
                            probe['lf_meta'] = c_info
                            if 'lf_data' in probe:
                                probe['lf_data'].append(c)
                            else:
                                probe['lf_data'] = [c]

                probes.append(probe)


