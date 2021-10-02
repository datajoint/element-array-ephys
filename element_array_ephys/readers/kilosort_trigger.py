import subprocess
import sys
import pathlib
import json
import re
import inspect


# import the spike sorting packages
try:
    from ecephys_spike_sorting.scripts.create_input_json import createInputJson
    from ecephys_spike_sorting.scripts.helpers import SpikeGLX_utils, log_from_json
except Exception as e:
    print(f'Error in loading "ecephys_spike_sorting" - {str(e)}')


class SGLXKilosortTrigger:
    """
    Triggering kilosort analysis for neuropixels data acquired
     from the SpikeGLX acquisition software
    Primarily calling routines specified from:
    https://github.com/jenniferColonell/ecephys_spike_sorting
    """

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

    _input_json_args = list(inspect.signature(createInputJson).parameters)

    def __init__(self, npx_input_dir: str, ks_output_dir: str,
                 params: dict, KS2ver: str,
                 run_CatGT=False,
                 ni_present=False,
                 ni_extract_string=None):

        self._npx_input_dir = pathlib.Path(npx_input_dir)

        self._ks_output_dir = pathlib.Path(ks_output_dir)
        self._ks_output_dir.mkdir(parents=True, exist_ok=True)

        self._params = params
        self._KS2ver = KS2ver
        self._run_CatGT = run_CatGT
        self._run_CatGT = run_CatGT
        self._default_catgt_params['ni_present'] = ni_present
        self._default_catgt_params['ni_extract_string'] = ni_extract_string or self._default_catgt_params['ni_extract_string']

        self._json_directory = self._ks_output_dir / 'json_configs'
        self._json_directory.mkdir(parents=True, exist_ok=True)

        self._CatGT_finished = False
        self._modules_finished = False

    def parse_input_filename(self):
        meta_filename = next(self._npx_input_dir.glob('*.ap.meta')).name
        match = re.search('(.*)_g(\d{1})_t(\d+|cat)\.imec(\d?)\.ap\.meta', meta_filename)
        session_str, gate_str, trigger_str, probe_str = match.groups()
        return session_str, gate_str, trigger_str, probe_str or '0'

    def generate_CatGT_input_json(self):
        if not self._run_CatGT:
            print('run_CatGT is set to False, skipping...')
            return

        session_str, gate_str, _, probe_str = self.parse_input_filename()

        first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(
            'start,end', probe_str, gate_str, self._npx_input_dir.as_posix())
        trigger_str = repr(first_trig) + ',' + repr(last_trig)

        self._catGT_input_json = self._json_directory / f'{session_str}{probe_str}_CatGT-input.json'

        catgt_params = {k: self._params.get(k, v)
                        for k, v in self._default_catgt_params.items()}

        ni_present = catgt_params.pop('ni_present')
        ni_extract_string = catgt_params.pop('ni_extract_string')

        catgt_params['catGT_stream_string'] = '-ap -ni' if ni_present else '-ap'
        sync_extract = '-SY=' + probe_str + ',-1,6,500'
        extract_string = sync_extract + (f' {ni_extract_string}' if ni_present else '')
        catgt_params['catGT_cmd_string'] += f' {extract_string}'

        input_meta_fullpath, continuous_file = self._get_raw_data_filepaths()

        createInputJson(self._catGT_input_json.as_posix(),
                        KS2ver=self._KS2ver,
                        npx_directory=self._npx_input_dir.as_posix(),
                        spikeGLX_data=True,
                        catGT_run_name=session_str,
                        gate_string=gate_str,
                        trigger_string=trigger_str,
                        probe_string=probe_str,
                        continuous_file=continuous_file.as_posix(),
                        input_meta_path=input_meta_fullpath.as_posix(),
                        extracted_data_directory=self._ks_output_dir.parent.as_posix(),
                        kilosort_output_directory=self._ks_output_dir.as_posix(),
                        **{k: v for k, v in catgt_params.items() if k in self._input_json_args}
                        )

    def run_CatGT(self, force_rerun=False):
        if self._run_CatGT and (not self._CatGT_finished or force_rerun):
            self.generate_CatGT_input_json()

            print('---- Running CatGT ----')
            catGT_input_json = self._catGT_input_json.as_posix()
            catGT_output_json = catGT_input_json.replace('CatGT-input.json', 'CatGT-output.json')

            command = (sys.executable
                       + " -W ignore -m ecephys_spike_sorting.modules."
                       + 'catGT_helper' + " --input_json " + catGT_input_json
                       + " --output_json " + catGT_output_json)
            subprocess.check_call(command.split(' '))

            self._CatGT_finished = True

    def generate_modules_input_json(self):
        session_str, gate_str, _, probe_str = self.parse_input_filename()
        self._module_input_json = self._json_directory / f'{session_str}_imec{probe_str}-input.json'

        input_meta_fullpath, continuous_file = self._get_raw_data_filepaths()

        ks_params = {k if k.startswith('ks_') else f'ks_{k}': str(v) if isinstance(v, list) else v
                     for k, v in self._params.items()}

        createInputJson(self._module_input_json.as_posix(),
                        KS2ver=self._KS2ver,
                        npx_directory=self._npx_input_dir.as_posix(),
                        spikeGLX_data=True,
                        continuous_file=continuous_file.as_posix(),
                        input_meta_path=input_meta_fullpath.as_posix(),
                        extracted_data_directory=self._ks_output_dir.parent.as_posix(),
                        kilosort_output_directory=self._ks_output_dir.as_posix(),
                        ks_make_copy=True,
                        noise_template_use_rf=self._params.get('noise_template_use_rf', True),
                        c_Waves_snr_um=self._params.get('c_Waves_snr_um', 160),
                        qm_isi_thresh=self._params.get('refPerMS', 2.0) / 1000,
                        **{k: v for k, v in ks_params.items() if k in self._input_json_args}
                        )

    def run_modules(self):
        if self._run_CatGT and not self._CatGT_finished:
            self.run_CatGT()

        print('---- Running Modules ----')
        self.generate_modules_input_json()
        module_input_json = self._module_input_json.as_posix()

        for module in self._modules:
            module_output_json = module_input_json.replace('-input.json',
                                                           module + '-output.json')

            command = (sys.executable
                       + " -W ignore -m ecephys_spike_sorting.modules." + module
                       + " --input_json " + module_input_json
                       + " --output_json " + module_output_json)
            subprocess.check_call(command.split(' '))

    def _get_raw_data_filepaths(self):
        session_str, gate_str, _, probe_str = self.parse_input_filename()

        if self._CatGT_finished:
            catGT_dest = self._ks_output_dir.parent
            run_str = session_str + '_g' + gate_str
            run_folder = 'catgt_' + run_str
            prb_folder = run_str + '_imec' + probe_str
            data_directory = catGT_dest / run_folder / prb_folder
        else:
            data_directory = self._npx_input_dir

        meta_fp = next(data_directory.glob(f'{session_str}*.ap.meta'))
        bin_fp = next(data_directory.glob(f'{session_str}*.ap.bin'))

        return meta_fp, bin_fp