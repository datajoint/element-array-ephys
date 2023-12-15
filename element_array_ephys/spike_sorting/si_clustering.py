"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the
"spikeinterface" pipeline.
Spikeinterface developed by Alessio Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, and Matthias Hennig (https://github.com/SpikeInterface)

The DataJoint pipeline currently incorporated Spikeinterfaces approach of running Kilosort using a container

The follow pipeline features intermediary tables:
1. PreProcessing - for preprocessing steps (no GPU required)
    - create recording extractor and link it to a probe
    - bandpass filtering
    - common mode referencing
2. SIClustering - kilosort (MATLAB) - requires GPU and docker/singularity containers
    - supports kilosort 2.0, 2.5 or 3.0 (https://github.com/MouseLand/Kilosort.git)
3. PostProcessing - for postprocessing steps (no GPU required)
    - create waveform extractor object
    - extract templates, waveforms and snrs
    - quality_metrics
"""

import datajoint as dj
import os
from element_array_ephys import get_logger
from datetime import datetime

from element_interface.utils import find_full_path
from element_array_ephys.readers import (
    spikeglx,
    kilosort_triggering,
)
import element_array_ephys.probe as probe

import spikeinterface as si
from element_interface.utils import find_full_path, find_root_directory
from spikeinterface import sorters

from element_array_ephys import get_logger, probe, readers

from .preprocessing import (
    mimic_catGT,
    mimic_IBLdestriping,
    mimic_IBLdestriping_modified,
)

log = get_logger(__name__)

schema = dj.schema()

ephys = None

_supported_kilosort_versions = [
    "kilosort2",
    "kilosort2.5",
    "kilosort3",
]


def activate(
    schema_name,
    *,
    ephys_module,
    create_schema=True,
    create_tables=True,
):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `spike_sorting` schema
        :param ephys_module: the activated ephys element for which this `spike_sorting` schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    global ephys
    ephys = ephys_module
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=ephys.__dict__,
    )


@schema
class PreProcessing(dj.Imported):
    """A table to handle preprocessing of each clustering task."""

    definition = """
    -> ephys.ClusteringTask
    ---
    recording_filename: varchar(30)     # filename where recording object is saved to
    params: longblob           # finalized parameterset for this run
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    @property
    def key_source(self):
        return ((
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & 'clustering_method in ("kilosort2", "kilosort2.5", "kilosort3")'
        ) - ephys.Clustering).proj()

    def make(self, key):
        """Triggers or imports clustering analysis."""
        execution_time = datetime.utcnow()

        task_mode, output_dir = (ephys.ClusteringTask & key).fetch1(
            "task_mode", "clustering_output_dir"
        )

        assert task_mode == "trigger", 'Supporting "trigger" task_mode only'

        if not output_dir:
            output_dir = ephys.ClusteringTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update clustering_output_dir
            ephys.ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )

        kilosort_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        acq_software, clustering_method, params = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("acq_software", "clustering_method", "params")

        assert (
            clustering_method in _supported_kilosort_versions
        ), f'Clustering_method "{clustering_method}" is not supported'

        if clustering_method.startswith("kilosort2.5"):
            sorter_name = "kilosort2_5"
        else:
            sorter_name = clustering_method
        # add additional probe-recording and channels details into `params`
        # params = {**params, **ephys.get_recording_channels_details(key)}
        # params["fs"] = params["sample_rate"]

        # default_params = si.full.get_default_sorter_params(sorter_name)
        # preprocess_list = params.pop("PreProcessing_params")

        if acq_software == "SpikeGLX":
            # sglx_session_full_path = find_full_path(ephys.get_ephys_root_data_dir(),ephys.get_session_directory(key))
            sglx_filepath = ephys.get_spikeglx_meta_filepath(key)

            # Create SI recording extractor object
            stream_name = sglx_filepath.stem.split(".", 1)[1]
            sglx_si_recording = si.extractors.read_spikeglx(
                folder_path=sglx_filepath.parent,
                stream_name=stream_name,
                stream_id=stream_name,
            )

            channels_details = ephys.get_recording_channels_details(key)
            xy_coords = [
                list(i)
                for i in zip(channels_details["x_coords"], channels_details["y_coords"])
            ]

            # Create SI probe object
            si_probe = pi.Probe(ndim=2, si_units="um")
            si_probe.set_contacts(
                positions=xy_coords, shapes="square", shape_params={"width": 12}
            )
            si_probe.create_auto_shape(probe_type="tip")
            si_probe.set_device_channel_indices(channels_details["channel_ind"])
            sglx_si_recording.set_probe(probe=si_probe)

            # # run preprocessing and save results to output folder
            # sglx_si_recording_filtered = si.preprocessing.bandpass_filter(
            #     sglx_si_recording, freq_min=300, freq_max=6000
            # )
            # sglx_recording_cmr = si.preprocessing.common_reference(sglx_si_recording_filtered, reference="global", operator="median")
            sglx_si_recording = mimic_catGT(sglx_si_recording)
            save_file_name = "si_recording.pkl"
            save_file_path = kilosort_dir / save_file_name
            sglx_si_recording.dump_to_pickle(file_path=save_file_path)

        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            oe_session_full_path = find_full_path(
                ephys.get_ephys_root_data_dir(), ephys.get_session_directory(key)
            )

            assert len(oe_probe.recording_info["recording_files"]) == 1
            stream_name = os.path.split(oe_probe.recording_info["recording_files"][0])[
                1
            ]

            # Create SI recording extractor object
            # oe_si_recording = si.extractors.OpenEphysBinaryRecordingExtractor(folder_path=oe_full_path, stream_name=stream_name)
            oe_si_recording = si.extractors.read_openephys(
                folder_path=oe_session_full_path, stream_name=stream_name
            )

            channels_details = ephys.get_recording_channels_details(key)
            xy_coords = [
                list(i)
                for i in zip(channels_details["x_coords"], channels_details["y_coords"])
            ]

            # Create SI probe object
            si_probe = pi.Probe(ndim=2, si_units="um")
            si_probe.set_contacts(
                positions=xy_coords, shapes="square", shape_params={"width": 12}
            )
            si_probe.create_auto_shape(probe_type="tip")
            si_probe.set_device_channel_indices(channels_details["channel_ind"])
            oe_si_recording.set_probe(probe=si_probe)

            # run preprocessing and save results to output folder
            # # Switch case to allow for specified preprocessing steps
            # oe_si_recording_filtered = si.preprocessing.bandpass_filter(
            #     oe_si_recording, freq_min=300, freq_max=6000
            # )
            # oe_recording_cmr = si.preprocessing.common_reference(
            #     oe_si_recording_filtered, reference="global", operator="median"
            # )
            oe_si_recording = mimic_IBLdestriping(oe_si_recording)
            save_file_name = "si_recording.pkl"
            save_file_path = kilosort_dir / save_file_name
            oe_si_recording.dump_to_pickle(file_path=save_file_path)

        self.insert1(
            {
                **key,
                "recording_filename": save_file_name,
                "params": params,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )


@schema
class SIClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> PreProcessing
    ---
    sorting_filename: varchar(30)   # filename of saved sorting object
    execution_time: datetime    # datetime of the start of this step
    execution_duration: float   # (hour) execution duration
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        kilosort_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        acq_software, clustering_method = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("acq_software", "clustering_method")

        params = (PreProcessing & key).fetch1("params")
        recording_filename = (PreProcessing & key).fetch1("recording_filename")

        if acq_software == "SpikeGLX":
            # sglx_probe = ephys.get_openephys_probe_data(key)
            recording_fullpath = kilosort_dir / recording_filename
            # sglx_si_recording = si.extractors.load_from_folder(recording_file)
            sglx_si_recording = si.core.load_extractor(recording_fullpath)
            # assert len(oe_probe.recording_info["recording_files"]) == 1

            ## Assume that the worker process will trigger this sorting step
            # - Will need to store/load the sorter_name, sglx_si_recording object etc.
            # - Store in shared EC2 space accessible by all containers (needs to be mounted)
            # - Load into the cloud init script, and
            # - Option A: Can call this function within a separate container within spike_sorting_worker
            if clustering_method.startswith("kilosort2.5"):
                sorter_name = "kilosort2_5"
            else:
                sorter_name = clustering_method
            sorting_kilosort = si.full.run_sorter(
                sorter_name=sorter_name,
                recording=sglx_si_recording,
                output_folder=kilosort_dir,
                docker_image=f"spikeinterface/{sorter_name}-compiled-base:latest",
                **params,
            )
            sorting_save_path = kilosort_dir / "sorting_kilosort.pkl"
            sorting_kilosort.dump_to_pickle(sorting_save_path)
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            oe_si_recording = si.core.load_extractor(recording_fullpath)
            assert len(oe_probe.recording_info["recording_files"]) == 1
            if clustering_method.startswith("kilosort2.5"):
                sorter_name = "kilosort2_5"
            else:
                sorter_name = clustering_method
            sorting_kilosort = si.full.run_sorter(
                sorter_name=sorter_name,
                recording=oe_si_recording,
                output_folder=kilosort_dir,
                docker_image=f"spikeinterface/{sorter_name}-compiled-base:latest",
                **params,
            )
            sorting_save_path = kilosort_dir / "sorting_kilosort.pkl"
            sorting_kilosort.dump_to_pickle(sorting_save_path)

        self.insert1(
            {
                **key,
                "sorting_filename": list(sorting_save_path.parts)[-1],
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )


@schema
class PostProcessing(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> SIClustering
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        kilosort_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        acq_software, clustering_method = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("acq_software", "clustering_method")

        params = (PreProcessing & key).fetch1("params")

        if acq_software == "SpikeGLX":
            recording_filename = (PreProcessing & key).fetch1("recording_filename")
            sorting_file = kilosort_dir / "sorting_kilosort"
            filtered_recording_file = kilosort_dir / recording_filename
            sglx_si_recording_filtered = si.core.load_extractor(recording_file)
            sorting_kilosort = si.core.load_extractor(sorting_file)

            we_kilosort = si.full.WaveformExtractor.create(
                sglx_si_recording_filtered,
                sorting_kilosort,
                "waveforms",
                remove_if_exists=True,
            )
            we_kilosort.set_params(ms_before=3.0, ms_after=4.0, max_spikes_per_unit=500)
            we_kilosort.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
            unit_id0 = sorting_kilosort.unit_ids[0]
            waveforms = we_kilosort.get_waveforms(unit_id0)
            template = we_kilosort.get_template(unit_id0)
            snrs = si.full.compute_snrs(we_kilosort)

            # QC Metrics
            (
                si_violations_ratio,
                isi_violations_rate,
                isi_violations_count,
            ) = si.full.compute_isi_violations(we_kilosort, isi_threshold_ms=1.5)
            metrics = si.full.compute_quality_metrics(
                we_kilosort,
                metric_names=[
                    "firing_rate",
                    "snr",
                    "presence_ratio",
                    "isi_violation",
                    "num_spikes",
                    "amplitude_cutoff",
                    "amplitude_median",
                    # "sliding_rp_violation",
                    "rp_violation",
                    "drift",
                ],
            )
            si.exporters.export_report(
                we_kilosort, kilosort_dir, n_jobs=-1, chunk_size=30000
            )
            # ["firing_rate","snr","presence_ratio","isi_violation",
            # "number_violation","amplitude_cutoff","isolation_distance","l_ratio","d_prime","nn_hit_rate",
            # "nn_miss_rate","silhouette_core","cumulative_drift","contamination_rate"])
            we_savedir = kilosort_dir / "we_kilosort"
            we_kilosort.save(we_savedir, n_jobs=-1, chunk_size=30000)

        elif acq_software == "Open Ephys":
            sorting_file = kilosort_dir / "sorting_kilosort"
            recording_file = kilosort_dir / "sglx_recording_cmr.json"
            oe_si_recording = si.core.load_extractor(recording_file)
            sorting_kilosort = si.core.load_extractor(sorting_file)

            we_kilosort = si.full.WaveformExtractor.create(
                oe_si_recording, sorting_kilosort, "waveforms", remove_if_exists=True
            )
            we_kilosort.set_params(ms_before=3.0, ms_after=4.0, max_spikes_per_unit=500)
            we_kilosort.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
            unit_id0 = sorting_kilosort.unit_ids[0]
            waveforms = we_kilosort.get_waveforms(unit_id0)
            template = we_kilosort.get_template(unit_id0)
            snrs = si.full.compute_snrs(we_kilosort)

            # QC Metrics
            # Apply waveform extractor extensions
            _ = si.full.compute_spike_locations(we_kilosort)
            _ = si.full.compute_spike_amplitudes(we_kilosort)
            _ = si.full.compute_unit_locations(we_kilosort)
            _ = si.full.compute_template_metrics(we_kilosort)
            _ = si.full.compute_noise_levels(we_kilosort)
            _ = si.full.compute_principal_components(we_kilosort)
            _ = si.full.compute_drift_metrics(we_kilosort)
            _ = si.full.compute_tempoate_similarity(we_kilosort)
            (
                isi_violations_ratio,
                isi_violations_count,
            ) = si.full.compute_isi_violations(we_kilosort, isi_threshold_ms=1.5)
            (isi_histograms, bins) = si.full.compute_isi_histograms(we_kilosort)
            metrics = si.full.compute_quality_metrics(
                we_kilosort,
                metric_names=[
                    "firing_rate",
                    "snr",
                    "presence_ratio",
                    "isi_violation",
                    "num_spikes",
                    "amplitude_cutoff",
                    "amplitude_median",
                    # "sliding_rp_violation",
                    "rp_violation",
                    "drift",
                ],
            )
            si.exporters.export_report(
                we_kilosort, kilosort_dir, n_jobs=-1, chunk_size=30000
            )
            we_savedir = kilosort_dir / "we_kilosort"
            we_kilosort.save(we_savedir, n_jobs=-1, chunk_size=30000)

            metrics_savefile = kilosort_dir / "metrics.csv"
            metrics.to_csv(metrics_savefile)

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )

        # all finished, insert this `key` into ephys.Clustering
        ephys.Clustering.insert1(
            {**key, "clustering_time": datetime.utcnow()}, allow_direct_insert=True
        )


# def runPreProcessList(preprocess_list, recording):
#     # If else
#     # need to figure out ordering
#     if preprocess_list["Filter"]:
#         recording = si.preprocessing.FilterRecording(recording)
#     if preprocess_list["BandpassFilter"]:
#         recording = si.preprocessing.BandpassFilterRecording(recording)
#     if preprocess_list["HighpassFilter"]:
#         recording = si.preprocessing.HighpassFilterRecording(recording)
#     if preprocess_list["NormalizeByQuantile"]:
#         recording = si.preprocessing.NormalizeByQuantileRecording(recording)
#     if preprocess_list["Scale"]:
#         recording = si.preprocessing.ScaleRecording(recording)
#     if preprocess_list["Center"]:
#         recording = si.preprocessing.CenterRecording(recording)
#     if preprocess_list["ZScore"]:
#         recording = si.preprocessing.ZScoreRecording(recording)
#     if preprocess_list["Whiten"]:
#         recording = si.preprocessing.WhitenRecording(recording)
#     if preprocess_list["CommonReference"]:
#         recording = si.preprocessing.CommonReferenceRecording(recording)
#     if preprocess_list["PhaseShift"]:
#         recording = si.preprocessing.PhaseShiftRecording(recording)
#     elif preprocess_list["Rectify"]:
#         recording = si.preprocessing.RectifyRecording(recording)
#     elif preprocess_list["Clip"]:
#         recording = si.preprocessing.ClipRecording(recording)
#     elif preprocess_list["BlankSaturation"]:
#         recording = si.preprocessing.BlankSaturationRecording(recording)
#     elif preprocess_list["RemoveArtifacts"]:
#         recording = si.preprocessing.RemoveArtifactsRecording(recording)
#     elif preprocess_list["RemoveBadChannels"]:
#         recording = si.preprocessing.RemoveBadChannelsRecording(recording)
#     elif preprocess_list["ZeroChannelPad"]:
#         recording = si.preprocessing.ZeroChannelPadRecording(recording)
#     elif preprocess_list["DeepInterpolation"]:
#         recording = si.preprocessing.DeepInterpolationRecording(recording)
#     elif preprocess_list["Resample"]:
#         recording = si.preprocessing.ResampleRecording(recording)


def mimic_IBLdestriping_modified(recording):
    # From SpikeInterface Implementation (https://spikeinterface.readthedocs.io/en/latest/how_to/analyse_neuropixels.html)
    recording = si.full.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.full.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = recording.remove_channels(bad_channel_ids)
    recording = si.full.phase_shift(recording)
    recording = si.full.common_reference(
        recording, operator="median", reference="global"
    )
    return recording


def mimic_IBLdestriping(recording):
    # From International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022.
    recording = si.full.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.full.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = si.preprocessing.interpolate_bad_channels(bad_channel_ids)
    recording = si.full.phase_shift(recording)
    # For IBL destriping use highpass_spatial_filter used instead of common reference
    recording = si.full.highpass_spatial_filter(
        recording, operator="median", reference="global"
    )
    return recording


def mimic_catGT(sglx_recording):
    sglx_recording = si.full.phase_shift(sglx_recording)
    sglx_recording = si.full.common_reference(
        sglx_recording, operator="median", reference="global"
    )
    return sglx_recording


## Example SI parameter set
"""
{'detect_threshold': 6,
 'projection_threshold': [10, 4],
 'preclust_threshold': 8,
 'car': True,
 'minFR': 0.02,
 'minfr_goodchannels': 0.1,
 'nblocks': 5,
 'sig': 20,
 'freq_min': 150,
 'sigmaMask': 30,
 'nPCs': 3,
 'ntbuff': 64,
 'nfilt_factor': 4,
 'NT': None,
 'do_correction': True,
 'wave_length': 61,
 'keep_good_only': False,
 'PreProcessing_params': {'Filter': False,
  'BandpassFilter': True,
  'HighpassFilter': False,
  'NotchFilter': False,
  'NormalizeByQuantile': False,
  'Scale': False,
  'Center': False,
  'ZScore': False,
  'Whiten': False,
  'CommonReference': False,
  'PhaseShift': False,
  'Rectify': False,
  'Clip': False,
  'BlankSaturation': False,
  'RemoveArtifacts': False,
  'RemoveBadChannels': False,
  'ZeroChannelPad': False,
  'DeepInterpolation': False,
  'Resample': False}}
"""
