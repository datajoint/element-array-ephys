import inspect
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import scipy.io
from element_interface.utils import dict_to_uuid

# import the spike sorting packages
try:
    from ecephys_spike_sorting.modules.kilosort_helper.__main__ import (
        get_noise_channels,
    )
    from ecephys_spike_sorting.scripts.create_input_json import createInputJson
    from ecephys_spike_sorting.scripts.helpers import SpikeGLX_utils
except Exception as e:
    print(f'Warning: Failed loading "ecephys_spike_sorting" package - {str(e)}')

# import pykilosort package
try:
    import pykilosort
except Exception as e:
    print(f'Warning: Failed loading "pykilosort" package - {str(e)}')


class SGLXKilosortPipeline:
    """
    An object of SGLXKilosortPipeline manages the state of the Kilosort data processing pipeline
     for one Neuropixels probe in one recording session using the Spike GLX acquisition software.

    Primarily calling routines specified from:
    https://github.com/jenniferColonell/ecephys_spike_sorting
    """

    _modules = [
        "kilosort_helper",
        "kilosort_postprocessing",
        "noise_templates",
        "mean_waveforms",
        "quality_metrics",
    ]

    _default_catgt_params = {
        "catGT_car_mode": "gblcar",
        "catGT_loccar_min_um": 40,
        "catGT_loccar_max_um": 160,
        "catGT_cmd_string": "-prb_fld -out_prb_fld -gfix=0.4,0.10,0.02",
        "ni_present": False,
        "ni_extract_string": "-XA=0,1,3,500 -iXA=1,3,3,0  -XD=-1,1,50 -XD=-1,2,1.7 -XD=-1,3,5 -iXD=-1,3,5",
    }

    _input_json_args = list(inspect.signature(createInputJson).parameters)

    def __init__(
        self,
        npx_input_dir: str,
        ks_output_dir: str,
        params: dict,
        KS2ver: str,
        run_CatGT=False,
        ni_present=False,
        ni_extract_string=None,
    ):
        self._npx_input_dir = pathlib.Path(npx_input_dir)

        self._ks_output_dir = pathlib.Path(ks_output_dir)
        self._ks_output_dir.mkdir(parents=True, exist_ok=True)

        self._params = params
        self._KS2ver = KS2ver
        self._run_CatGT = run_CatGT
        self._run_CatGT = run_CatGT
        self._default_catgt_params["ni_present"] = ni_present
        self._default_catgt_params["ni_extract_string"] = (
            ni_extract_string or self._default_catgt_params["ni_extract_string"]
        )

        self._json_directory = self._ks_output_dir / "json_configs"
        self._json_directory.mkdir(parents=True, exist_ok=True)

        self._module_input_json = (
            self._json_directory / f"{self._npx_input_dir.name}-input.json"
        )
        self._module_logfile = (
            self._json_directory / f"{self._npx_input_dir.name}-run_modules-log.txt"
        )

        self._CatGT_finished = False
        self.ks_input_params = None
        self._modules_input_hash = None
        self._modules_input_hash_fp = None

    def parse_input_filename(self):
        meta_filename = next(self._npx_input_dir.glob("*.ap.meta")).name
        match = re.search("(.*)_g(\d)_t(\d+|cat)\.imec(\d?)\.ap\.meta", meta_filename)
        session_str, gate_str, trigger_str, probe_str = match.groups()
        return session_str, gate_str, trigger_str, probe_str or "0"

    def generate_CatGT_input_json(self):
        if not self._run_CatGT:
            print("run_CatGT is set to False, skipping...")
            return

        session_str, gate_str, trig_str, probe_str = self.parse_input_filename()

        first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(
            "start,end", probe_str, gate_str, self._npx_input_dir.as_posix()
        )
        trigger_str = repr(first_trig) + "," + repr(last_trig)

        self._catGT_input_json = (
            self._json_directory / f"{session_str}{probe_str}_CatGT-input.json"
        )

        catgt_params = {
            k: self._params.get(k, v) for k, v in self._default_catgt_params.items()
        }

        ni_present = catgt_params.pop("ni_present")
        ni_extract_string = catgt_params.pop("ni_extract_string")

        catgt_params["catGT_stream_string"] = "-ap -ni" if ni_present else "-ap"
        sync_extract = "-SY=" + probe_str + ",-1,6,500"
        extract_string = sync_extract + (f" {ni_extract_string}" if ni_present else "")
        catgt_params["catGT_cmd_string"] += f" {extract_string}"

        input_meta_fullpath, continuous_file = self._get_raw_data_filepaths()

        # create symbolic link to the actual data files - as CatGT expects files to follow a certain naming convention
        continuous_file_symlink = (
            continuous_file.parent
            / f"{session_str}_g{gate_str}"
            / f"{session_str}_g{gate_str}_imec{probe_str}"
            / f"{session_str}_g{gate_str}_t{trig_str}.imec{probe_str}.ap.bin"
        )
        continuous_file_symlink.parent.mkdir(parents=True, exist_ok=True)
        if continuous_file_symlink.exists():
            continuous_file_symlink.unlink()
        continuous_file_symlink.symlink_to(continuous_file)
        input_meta_fullpath_symlink = (
            input_meta_fullpath.parent
            / f"{session_str}_g{gate_str}"
            / f"{session_str}_g{gate_str}_imec{probe_str}"
            / f"{session_str}_g{gate_str}_t{trig_str}.imec{probe_str}.ap.meta"
        )
        input_meta_fullpath_symlink.parent.mkdir(parents=True, exist_ok=True)
        if input_meta_fullpath_symlink.exists():
            input_meta_fullpath_symlink.unlink()
        input_meta_fullpath_symlink.symlink_to(input_meta_fullpath)

        createInputJson(
            self._catGT_input_json.as_posix(),
            KS2ver=self._KS2ver,
            npx_directory=self._npx_input_dir.as_posix(),
            spikeGLX_data=True,
            catGT_run_name=session_str,
            gate_string=gate_str,
            trigger_string=trigger_str,
            probe_string=probe_str,
            continuous_file=continuous_file.as_posix(),
            input_meta_path=input_meta_fullpath.as_posix(),
            extracted_data_directory=self._ks_output_dir.as_posix(),
            kilosort_output_directory=self._ks_output_dir.as_posix(),
            kilosort_output_tmp=self._ks_output_dir.as_posix(),
            kilosort_repository=_get_kilosort_repository(self._KS2ver),
            **{k: v for k, v in catgt_params.items() if k in self._input_json_args},
        )

    def run_CatGT(self, force_rerun=False):
        if self._run_CatGT and (not self._CatGT_finished or force_rerun):
            self.generate_CatGT_input_json()

            print("---- Running CatGT ----")
            catGT_input_json = self._catGT_input_json.as_posix()
            catGT_output_json = catGT_input_json.replace(
                "CatGT-input.json", "CatGT-output.json"
            )

            command = (
                sys.executable
                + " -W ignore -m ecephys_spike_sorting.modules."
                + "catGT_helper"
                + " --input_json "
                + catGT_input_json
                + " --output_json "
                + catGT_output_json
            )
            subprocess.check_call(command.split(" "))

            self._CatGT_finished = True

    def generate_modules_input_json(self):
        session_str, _, _, probe_str = self.parse_input_filename()
        self._module_input_json = (
            self._json_directory / f"{session_str}_imec{probe_str}-input.json"
        )

        input_meta_fullpath, continuous_file = self._get_raw_data_filepaths()

        params = {}
        for k, v in self._params.items():
            value = str(v) if isinstance(v, list) else v
            if f"ks_{k}" in self._input_json_args:
                params[f"ks_{k}"] = value
            if k in self._input_json_args:
                params[k] = value

        self.ks_input_params = createInputJson(
            self._module_input_json.as_posix(),
            KS2ver=self._KS2ver,
            npx_directory=self._npx_input_dir.as_posix(),
            spikeGLX_data=True,
            continuous_file=continuous_file.as_posix(),
            input_meta_path=input_meta_fullpath.as_posix(),
            extracted_data_directory=self._ks_output_dir.as_posix(),
            kilosort_output_directory=self._ks_output_dir.as_posix(),
            kilosort_output_tmp=self._ks_output_dir.as_posix(),
            ks_make_copy=True,
            noise_template_use_rf=self._params.get("noise_template_use_rf", False),
            c_Waves_snr_um=self._params.get("c_Waves_snr_um", 160),
            qm_isi_thresh=self._params.get("refPerMS", 2.0) / 1000,
            kilosort_repository=_get_kilosort_repository(self._KS2ver),
            **params,
        )

        self._modules_input_hash = dict_to_uuid(dict(self._params, KS2ver=self._KS2ver))

    def run_modules(self, modules_to_run=None):
        if self._run_CatGT and not self._CatGT_finished:
            self.run_CatGT()

        print("---- Running Modules ----")
        self.generate_modules_input_json()
        module_input_json = self._module_input_json.as_posix()
        module_logfile = self._module_logfile.as_posix()

        modules = modules_to_run or self._modules

        for module in modules:
            module_status = self._get_module_status(module)
            if module_status["completion_time"] is not None:
                continue

            module_output_json = self._get_module_output_json_filename(module)
            command = (
                sys.executable
                + " -W ignore -m ecephys_spike_sorting.modules."
                + module
                + " --input_json "
                + module_input_json
                + " --output_json "
                + module_output_json
            )

            start_time = datetime.utcnow()
            self._update_module_status(
                {
                    module: {
                        "start_time": start_time,
                        "completion_time": None,
                        "duration": None,
                    }
                }
            )
            with open(module_logfile, "a") as f:
                subprocess.check_call(command.split(" "), stdout=f)
            completion_time = datetime.utcnow()
            self._update_module_status(
                {
                    module: {
                        "start_time": start_time,
                        "completion_time": completion_time,
                        "duration": (completion_time - start_time).total_seconds(),
                    }
                }
            )

        self._update_total_duration()

    def _get_raw_data_filepaths(self):
        session_str, gate_str, _, probe_str = self.parse_input_filename()

        if self._CatGT_finished:
            catGT_dest = self._ks_output_dir
            run_str = session_str + "_g" + gate_str
            run_folder = "catgt_" + run_str
            prb_folder = run_str + "_imec" + probe_str
            data_directory = catGT_dest / run_folder / prb_folder
        else:
            data_directory = self._npx_input_dir
        try:
            meta_fp = next(data_directory.glob(f"{session_str}*.ap.meta"))
            bin_fp = next(data_directory.glob(f"{session_str}*.ap.bin"))
        except StopIteration:
            raise RuntimeError(
                f"No ap meta/bin files found in {data_directory} - CatGT error?"
            )

        return meta_fp, bin_fp

    def _update_module_status(self, updated_module_status={}):
        if self._modules_input_hash is None:
            raise RuntimeError('"generate_modules_input_json()" not yet performed!')

        self._modules_input_hash_fp = (
            self._json_directory / f".{self._modules_input_hash}.json"
        )
        if self._modules_input_hash_fp.exists():
            with open(self._modules_input_hash_fp) as f:
                modules_status = json.load(f)
            modules_status = {**modules_status, **updated_module_status}
        else:
            # handle cases of processing rerun on different parameters (the hash changes)
            # delete outdated files
            [
                f.unlink()
                for f in self._json_directory.glob("*")
                if f.is_file() and f.name != self._module_input_json.name
            ]

            modules_status = {
                module: {"start_time": None, "completion_time": None, "duration": None}
                for module in self._modules
            }
        with open(self._modules_input_hash_fp, "w") as f:
            json.dump(modules_status, f, default=str)

    def _get_module_status(self, module):
        if self._modules_input_hash_fp is None:
            self._update_module_status()

        if self._modules_input_hash_fp.exists():
            with open(self._modules_input_hash_fp) as f:
                modules_status = json.load(f)
            if modules_status[module]["completion_time"] is None:
                # additional logic to read from the "-output.json" file for this module as well
                # handle cases where the module has finished successfully,
                # but the "_modules_input_hash_fp" is not updated (for whatever reason),
                # resulting in this module not registered as completed in the "_modules_input_hash_fp"
                module_output_json_fp = pathlib.Path(
                    self._get_module_output_json_filename(module)
                )
                if module_output_json_fp.exists():
                    with open(module_output_json_fp) as f:
                        module_run_output = json.load(f)
                    modules_status[module]["duration"] = module_run_output[
                        "execution_time"
                    ]
                    modules_status[module]["completion_time"] = datetime.strptime(
                        modules_status[module]["start_time"], "%Y-%m-%d %H:%M:%S.%f"
                    ) + timedelta(seconds=module_run_output["execution_time"])
            return modules_status[module]

        return {"start_time": None, "completion_time": None, "duration": None}

    def _get_module_output_json_filename(self, module):
        module_input_json = self._module_input_json.as_posix()
        module_output_json = module_input_json.replace(
            "-input.json",
            "-" + module + "-" + str(self._modules_input_hash) + "-output.json",
        )
        return module_output_json

    def _update_total_duration(self):
        with open(self._modules_input_hash_fp) as f:
            modules_status = json.load(f)
        cumulative_execution_duration = sum(
            v["duration"] or 0
            for k, v in modules_status.items()
            if k not in ("cumulative_execution_duration", "total_duration")
        )

        for m in self._modules:
            first_start_time = modules_status[m]["start_time"]
            if first_start_time is not None:
                break

        for m in self._modules[::-1]:
            last_completion_time = modules_status[m]["completion_time"]
            if last_completion_time is not None:
                break

        if first_start_time is None or last_completion_time is None:
            return

        total_duration = (
            datetime.strptime(
                last_completion_time,
                "%Y-%m-%d %H:%M:%S.%f",
            )
            - datetime.strptime(first_start_time, "%Y-%m-%d %H:%M:%S.%f")
        ).total_seconds()
        self._update_module_status(
            {
                "cumulative_execution_duration": cumulative_execution_duration,
                "total_duration": total_duration,
            }
        )


class OpenEphysKilosortPipeline:
    """
    An object of OpenEphysKilosortPipeline manages the state of the Kilosort data processing pipeline
     for one Neuropixels probe in one recording session using the Open Ephys acquisition software.

    Primarily calling routines specified from:
    https://github.com/jenniferColonell/ecephys_spike_sorting
    Which is based on `ecephys_spike_sorting` routines from Allen Institute
    https://github.com/AllenInstitute/ecephys_spike_sorting
    """

    _modules = [
        "depth_estimation",
        "median_subtraction",
        "kilosort_helper",
        "kilosort_postprocessing",
        "noise_templates",
        "mean_waveforms",
        "quality_metrics",
    ]

    _input_json_args = list(inspect.signature(createInputJson).parameters)

    def __init__(
        self, npx_input_dir: str, ks_output_dir: str, params: dict, KS2ver: str
    ):
        self._npx_input_dir = pathlib.Path(npx_input_dir)

        self._ks_output_dir = pathlib.Path(ks_output_dir)
        self._ks_output_dir.mkdir(parents=True, exist_ok=True)

        self._params = params
        self._KS2ver = KS2ver

        self._json_directory = self._ks_output_dir / "json_configs"
        self._json_directory.mkdir(parents=True, exist_ok=True)

        self._module_input_json = (
            self._json_directory / f"{self._npx_input_dir.name}-input.json"
        )
        self._module_logfile = (
            self._json_directory / f"{self._npx_input_dir.name}-run_modules-log.txt"
        )

        self.ks_input_params = None
        self._modules_input_hash = None
        self._modules_input_hash_fp = None

    def make_chanmap_file(self):
        continuous_file = self._npx_input_dir / "continuous.dat"
        self._chanmap_filepath = self._ks_output_dir / "chanMap.mat"

        _write_channel_map_file(
            channel_ind=self._params["channel_ind"],
            x_coords=self._params["x_coords"],
            y_coords=self._params["y_coords"],
            shank_ind=self._params["shank_ind"],
            connected=self._params["connected"],
            probe_name=self._params["probe_type"],
            ap_band_file=continuous_file.as_posix(),
            bit_volts=self._params["uVPerBit"],
            sample_rate=self._params["sample_rate"],
            save_path=self._chanmap_filepath.as_posix(),
            is_0_based=True,
        )

    def generate_modules_input_json(self):
        self.make_chanmap_file()

        continuous_file = self._get_raw_data_filepaths()

        lf_dir = self._npx_input_dir.as_posix()
        try:
            # old probe folder convention with 100.0, 100.1, 100.2, 100.3, etc.
            name, num = re.search(r"(.+\.)(\d)+$", lf_dir).groups()
        except AttributeError:
            # new probe folder convention with -AP or -LFP
            assert lf_dir.endswith("AP")
            lf_dir = re.sub("-AP$", "-LFP", lf_dir)
        else:
            lf_dir = f"{name}{int(num) + 1}"
        lf_file = pathlib.Path(lf_dir) / "continuous.dat"

        params = {}
        for k, v in self._params.items():
            value = str(v) if isinstance(v, list) else v
            if f"ks_{k}" in self._input_json_args:
                params[f"ks_{k}"] = value
            if k in self._input_json_args:
                params[k] = value

        self.ks_input_params = createInputJson(
            self._module_input_json.as_posix(),
            KS2ver=self._KS2ver,
            npx_directory=self._npx_input_dir.as_posix(),
            spikeGLX_data=False,
            continuous_file=continuous_file.as_posix(),
            lf_file=lf_file.as_posix(),
            extracted_data_directory=self._ks_output_dir.as_posix(),
            kilosort_output_directory=self._ks_output_dir.as_posix(),
            kilosort_output_tmp=self._ks_output_dir.as_posix(),
            ks_make_copy=True,
            noise_template_use_rf=self._params.get("noise_template_use_rf", False),
            use_C_Waves=False,
            c_Waves_snr_um=self._params.get("c_Waves_snr_um", 160),
            qm_isi_thresh=self._params.get("refPerMS", 2.0) / 1000,
            kilosort_repository=_get_kilosort_repository(self._KS2ver),
            chanMap_path=self._chanmap_filepath.as_posix(),
            **params,
        )

        self._modules_input_hash = dict_to_uuid(dict(self._params, KS2ver=self._KS2ver))

    def run_modules(self, modules_to_run=None):
        print("---- Running Modules ----")
        self.generate_modules_input_json()
        module_input_json = self._module_input_json.as_posix()
        module_logfile = self._module_logfile.as_posix()

        modules = modules_to_run or self._modules

        for module in modules:
            module_status = self._get_module_status(module)
            if module_status["completion_time"] is not None:
                continue

            if module == "median_subtraction":
                median_subtraction_duration = (
                    self._get_median_subtraction_duration_from_log()
                )
                if median_subtraction_duration is not None:
                    median_subtraction_status = self._get_module_status(
                        "median_subtraction"
                    )
                    median_subtraction_status["duration"] = median_subtraction_duration
                    median_subtraction_status["completion_time"] = datetime.strptime(
                        median_subtraction_status["start_time"], "%Y-%m-%d %H:%M:%S.%f"
                    ) + timedelta(seconds=median_subtraction_status["duration"])
                    self._update_module_status(
                        {"median_subtraction": median_subtraction_status}
                    )
                    continue

            module_output_json = self._get_module_output_json_filename(module)
            command = [
                sys.executable,
                "-W",
                "ignore",
                "-m",
                "ecephys_spike_sorting.modules." + module,
                "--input_json",
                module_input_json,
                "--output_json",
                module_output_json,
            ]

            start_time = datetime.utcnow()
            self._update_module_status(
                {
                    module: {
                        "start_time": start_time,
                        "completion_time": None,
                        "duration": None,
                    }
                }
            )
            with open(module_logfile, "a") as f:
                subprocess.check_call(command, stdout=f)
            completion_time = datetime.utcnow()
            self._update_module_status(
                {
                    module: {
                        "start_time": start_time,
                        "completion_time": completion_time,
                        "duration": (completion_time - start_time).total_seconds(),
                    }
                }
            )

        self._update_total_duration()

    def _get_raw_data_filepaths(self):
        raw_ap_fp = self._npx_input_dir / "continuous.dat"

        if "median_subtraction" not in self._modules:
            return raw_ap_fp

        # median subtraction step will overwrite original continuous.dat file with the corrected version
        # to preserve the original raw data - make a copy here and work on the copied version
        assert "depth_estimation" in self._modules
        continuous_file = self._ks_output_dir / "continuous.dat"
        if continuous_file.exists():
            if raw_ap_fp.stat().st_mtime == continuous_file.stat().st_mtime:
                return continuous_file
            else:
                if self._module_logfile.exists():
                    return continuous_file

        shutil.copy2(raw_ap_fp, continuous_file)
        return continuous_file

    def _update_module_status(self, updated_module_status={}):
        if self._modules_input_hash is None:
            raise RuntimeError('"generate_modules_input_json()" not yet performed!')

        self._modules_input_hash_fp = (
            self._json_directory / f".{self._modules_input_hash}.json"
        )
        if self._modules_input_hash_fp.exists():
            with open(self._modules_input_hash_fp) as f:
                modules_status = json.load(f)
            modules_status = {**modules_status, **updated_module_status}
        else:
            # handle cases of processing rerun on different parameters (the hash changes)
            # delete outdated files
            [
                f.unlink()
                for f in self._json_directory.glob("*")
                if f.is_file() and f.name != self._module_input_json.name
            ]

            modules_status = {
                module: {"start_time": None, "completion_time": None, "duration": None}
                for module in self._modules
            }
        with open(self._modules_input_hash_fp, "w") as f:
            json.dump(modules_status, f, default=str)

    def _get_module_status(self, module):
        if self._modules_input_hash_fp is None:
            self._update_module_status()

        if self._modules_input_hash_fp.exists():
            with open(self._modules_input_hash_fp) as f:
                modules_status = json.load(f)
            if modules_status[module]["completion_time"] is None:
                # additional logic to read from the "-output.json" file for this module as well
                # handle cases where the module has finished successfully,
                # but the "_modules_input_hash_fp" is not updated (for whatever reason),
                # resulting in this module not registered as completed in the "_modules_input_hash_fp"
                module_output_json_fp = pathlib.Path(
                    self._get_module_output_json_filename(module)
                )
                if module_output_json_fp.exists():
                    with open(module_output_json_fp) as f:
                        module_run_output = json.load(f)
                    modules_status[module]["duration"] = module_run_output[
                        "execution_time"
                    ]
                    modules_status[module]["completion_time"] = datetime.strptime(
                        modules_status[module]["start_time"], "%Y-%m-%d %H:%M:%S.%f"
                    ) + timedelta(seconds=module_run_output["execution_time"])
            return modules_status[module]

        return {"start_time": None, "completion_time": None, "duration": None}

    def _get_module_output_json_filename(self, module):
        module_input_json = self._module_input_json.as_posix()
        module_output_json = module_input_json.replace(
            "-input.json",
            "-" + module + "-" + str(self._modules_input_hash) + "-output.json",
        )
        return module_output_json

    def _update_total_duration(self):
        with open(self._modules_input_hash_fp) as f:
            modules_status = json.load(f)
        cumulative_execution_duration = sum(
            v["duration"] or 0
            for k, v in modules_status.items()
            if k not in ("cumulative_execution_duration", "total_duration")
        )

        for m in self._modules:
            first_start_time = modules_status[m]["start_time"]
            if first_start_time is not None:
                break

        for m in self._modules[::-1]:
            last_completion_time = modules_status[m]["completion_time"]
            if last_completion_time is not None:
                break

        if first_start_time is None or last_completion_time is None:
            return

        total_duration = (
            datetime.strptime(
                last_completion_time,
                "%Y-%m-%d %H:%M:%S.%f",
            )
            - datetime.strptime(first_start_time, "%Y-%m-%d %H:%M:%S.%f")
        ).total_seconds()
        self._update_module_status(
            {
                "cumulative_execution_duration": cumulative_execution_duration,
                "total_duration": total_duration,
            }
        )

    def _get_median_subtraction_duration_from_log(self):
        raw_ap_fp = self._npx_input_dir / "continuous.dat"
        continuous_file = self._ks_output_dir / "continuous.dat"
        if raw_ap_fp.stat().st_mtime < continuous_file.stat().st_mtime:
            # if the copied continuous.dat was actually modified,
            # median_subtraction may have been completed - let's check
            if self._module_logfile.exists():
                with open(self._module_logfile, "r") as f:
                    previous_line = ""
                    for line in f.readlines():
                        if line.startswith(
                            "ecephys spike sorting: median subtraction module"
                        ) and previous_line.startswith("Total processing time:"):
                            # regex to search for the processing duration - a float value
                            duration = int(
                                re.search("\d+\.?\d+", previous_line).group()
                            )
                            return duration
                        previous_line = line


def run_pykilosort(
    continuous_file,
    kilosort_output_directory,
    params,
    channel_ind,
    x_coords,
    y_coords,
    shank_ind,
    connected,
    sample_rate,
):
    dat_path = pathlib.Path(continuous_file)

    probe = pykilosort.Bunch()
    channel_count = len(channel_ind)
    probe.Nchan = channel_count
    probe.NchanTOT = channel_count
    probe.chanMap = np.arange(0, channel_count, dtype="int")
    probe.xc = x_coords
    probe.yc = y_coords
    probe.kcoords = shank_ind

    pykilosort.run(
        dat_path=continuous_file,
        dir_path=dat_path.parent,
        output_dir=kilosort_output_directory,
        probe=probe,
        params=params,
        n_channels=probe.Nchan,
        dtype=np.int16,
        sample_rate=sample_rate,
    )


def _get_kilosort_repository(KS2ver):
    """
    Get the path to where the kilosort package is installed at, assuming it can be found
    as environment variable named "kilosort_repository"
    Modify this path according to the KSVer used
    """
    ks_repo = pathlib.Path(os.getenv("kilosort_repository"))
    assert ks_repo.exists()
    assert ks_repo.stem.startswith("Kilosort")

    ks_repo = ks_repo.parent / f"Kilosort-{KS2ver}"
    assert ks_repo.exists()

    return ks_repo.as_posix()


def _write_channel_map_file(
    *,
    channel_ind,
    x_coords,
    y_coords,
    shank_ind,
    connected,
    probe_name,
    ap_band_file,
    bit_volts,
    sample_rate,
    save_path,
    is_0_based=True,
):
    """
    Write channel map into .mat file in 1-based indexing format (MATLAB style)
    """

    assert (
        len(channel_ind)
        == len(x_coords)
        == len(y_coords)
        == len(shank_ind)
        == len(connected)
    )

    if is_0_based:
        channel_ind = channel_ind + 1
        shank_ind = shank_ind + 1

    channel_count = len(channel_ind)
    chanMap0ind = np.arange(0, channel_count, dtype="float64")
    chanMap0ind = chanMap0ind.reshape((channel_count, 1))
    chanMap = chanMap0ind + 1

    # channels to exclude
    mask = get_noise_channels(ap_band_file, channel_count, sample_rate, bit_volts)
    connected = np.where(mask is False, 0, connected)

    mdict = {
        "chanMap": chanMap,
        "chanMap0ind": chanMap0ind,
        "connected": connected,
        "name": probe_name,
        "xcoords": x_coords,
        "ycoords": y_coords,
        "shankInd": shank_ind,
        "kcoords": shank_ind,
        "fs": sample_rate,
    }

    scipy.io.savemat(save_path, mdict)
