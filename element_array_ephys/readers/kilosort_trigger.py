import subprocess
import sys
import pathlib
import json
import re


class SGLXKilosortTrigger:
    """
    Triggering kilosort analysis for neuropixels data acquired
     from the SpikeGLX acquisition software
    Primarily calling routines specified from:
    https://github.com/jenniferColonell/ecephys_spike_sorting
    """

    _required_packages_paths = {
        'ecephys_directory': None,
        'kilosort_repository': None,
        'npy_matlab_repository': None,
        'catGTPath': None,
        'tPrime_path': None,
        'cWaves_path': None,
        'kilosort_output_tmp': None
    }

    _modules = ['kilosort_helper',
                'kilosort_postprocessing',
                'noise_templates',
                'mean_waveforms',
                'quality_metrics']

    _default_catgt_params = {
        'catGT_car_mode': 'gblcar',
        'catGT_loccar_min_um': 40,
        'catGT_loccar_max_um': 160,
        'catGT_cmd_string': '-prb_fld -out_prb_fld -aphipass=300 -gfix=0.4,0.10,0.02 -tshift',
        'ni_present': False,
        'ni_extract_string': '-XA=0,1,3,500 -iXA=1,3,3,0  -XD=-1,1,50 -XD=-1,2,1.7 -XD=-1,3,5 -iXD=-1,3,5'
    }

    def __init__(self, npx_input_dir: str, ks_output_dir: str,
                 params: dict, KS2ver: str, run_CatGT=False):

        from ecephys_spike_sorting.scripts.create_input_json import createInputJson
        from ecephys_spike_sorting.scripts.helpers import SpikeGLX_utils

        self._npx_input_dir = pathlib.Path(npx_input_dir)

        self._ks_output_dir = pathlib.Path(ks_output_dir)
        self._ks_output_dir.mkdir(parents=True, exist_ok=True)

        self._params = params
        self._KS2ver = KS2ver
        self._run_CatGT = run_CatGT

        self._json_directory = self._ks_output_dir / 'json_configs'
        self._json_directory.mkdir(parents=True, exist_ok=True)

        self._CatGT_finished = None
        self._modules_finished = None

    def parse_input_filename(self):
        meta_filename = next(self._npx_input_dir.globl('*.ap.meta')).name
        match = re.search('(.*)_g(\d{1})_t(\d+|cat)\.imec(\d?)\.ap\.meta', meta_filename)
        session_str, gate_str, trigger_str, probe_str = match.groups()
        return session_str, gate_str, trigger_str, probe_str or '0'

    def generate_CatGT_input_json(self):
        session_str, gate_str, trigger_str, probe_str = self.parse_input_filename()

        first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(
            trigger_str, probe_str, gate_str, self._npx_input_dir.as_posix())
        trigger_str = repr(first_trig) + ',' + repr(last_trig)

        self._catGT_input_json = self._json_directory / f'{session_str}{probe_str}_CatGT-input.json'

        catgt_params = {k: self._params.get(k, v)
                        for k, v in self._default_catgt_params.items()}
        catgt_params['catGT_stream_string'] = '-ap -ni' if catgt_params['ni_present'] else '-ap'
        sync_extract = '-SY=' + probe_str + ',-1,6,500'
        extract_string = sync_extract + (' ' + catgt_params['ni_extract_string'] if catgt_params['ni_present'] else '')
        catgt_params['catGT_cmd_string'] += ' ' + extract_string

        createInputJson(self._catGT_input_json.as_posix(),
                        **self._required_packages_paths,
                        KS2ver=self._KS2ver,
                        npx_directory=self._npx_input_dir.as_posix(),
                        spikeGLX_data=True,
                        catGT_run_name=session_str,
                        gate_string=gate_str,
                        trigger_string=trigger_str,
                        probe_string=probe_str,
                        **catgt_params,
                        extracted_data_directory=self._ks_output_dir.parent.as_posix()
                        )

    def run_CatGT(self, force_rerun=False):
        if self._run_CatGT and (not self._CatGT_finished or force_rerun):
            self.generate_CatGT_input_json()

            catGT_input_json = self._catGT_input_json.as_posix()
            catGT_output_json = catGT_input_json.replace('CatGT-input.json', 'CatGT-output.json')

            command = (sys.executable
                       + " -W ignore -m ecephys_spike_sorting.modules."
                       + 'catGT_helper' + " --input_json " + catGT_input_json
                       + " --output_json " + catGT_output_json)
            subprocess.check_call(command.split(' '))

            self._CatGT_finished = True

    def generate_modules_input_json(self):
        session_str, gate_str, trigger_str, probe_str = self.parse_input_filename()
        self._module_input_json = self._json_directory / f'{session_str}_imec{probe_str}-input.json'

        if self._CatGT_finished:
            catGT_dest = self._ks_output_dir.parent
            run_str = session_str + '_g' + gate_str
            run_folder = 'catgt_' + run_str
            prb_folder = run_str + '_imec' + probe_str
            data_directory = catGT_dest / run_folder / prb_folder
        else:
            data_directory = self._npx_input_dir

        continuous_file = next(data_directory.glob(f'{session_str}*.ap.bin'))
        input_meta_fullpath = next(data_directory.glob(f'{session_str}*.ap.meta'))

        ks_params = {k if k.startswith('ks_') else f'ks_{k}': v
                     for k, v in self._params.items()}

        createInputJson(self._module_input_json.as_posix(),
                        **self._required_packages_paths,
                        KS2ver=self._KS2ver,
                        npx_directory=self._npx_input_dir.as_posix(),
                        continuous_file=continuous_file.as_posix(),
                        spikeGLX_data=True,
                        input_meta_path=input_meta_fullpath.as_posix(),
                        kilosort_output_directory=self._ks_output_dir.as_posix(),
                        ks_make_copy=True,
                        **ks_params,
                        noise_template_use_rf=self._params.get('noise_template_use_rf', True),
                        c_Waves_snr_um=self._params.get('c_Waves_snr_um', 160),
                        qm_isi_thresh=self._params.get('refPerMS', 2.0) / 1000
                        )

    def run_modules(self):
        if self._run_CatGT and not self._CatGT_finished:
            self.run_CatGT()

        module_input_json = self._module_input_json.as_posix()

        for module in self._modules:
            module_output_json = module_input_json.replace('-input.json',
                                                           module + '-output.json')

            command = (sys.executable
                       + " -W ignore -m ecephys_spike_sorting.modules." + module
                       + " --input_json " + module_input_json
                       + " --output_json " + module_output_json)
            subprocess.check_call(command.split(' '))
