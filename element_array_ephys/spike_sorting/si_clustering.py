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

from datetime import datetime

import datajoint as dj
import pandas as pd
import probeinterface as pi
import spikeinterface as si
from element_interface.utils import find_full_path, find_root_directory
from spikeinterface import exporters, qualitymetrics, sorters

from element_array_ephys import get_logger, probe, readers

from .preprocessing import (
    mimic_catGT,
    mimic_IBLdestriping,
    mimic_IBLdestriping_modified,
)

log = get_logger(__name__)

schema = dj.schema()

ephys = None


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


SI_SORTERS = [s.replace("_", ".") for s in si.sorters.sorter_dict.keys()]

SI_READERS = {
    "Open Ephys": si.extractors.read_openephys,
    "SpikeGLX": si.extractors.read_spikeglx,
    "Intan": si.extractors.read_intan,
}


@schema
class PreProcessing(dj.Imported):
    """A table to handle preprocessing of each clustering task. The output will be serialized and stored as a si_recording.pkl in the output directory."""

    definition = """
    -> ephys.ClusteringTask
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    """

    @property
    def key_source(self):
        return (
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & f"clustering_method in {tuple(SI_SORTERS)}"
        ) - ephys.Clustering

    def make(self, key):
        """Triggers or imports clustering analysis."""
        execution_time = datetime.utcnow()

        # Set the output directory
        acq_software, output_dir = (
            ephys.ClusteringTask * ephys.EphysRecording & key
        ).fetch1("acq_software", "clustering_output_dir")

        if not output_dir:
            output_dir = ephys.ClusteringTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update clustering_output_dir
            ephys.ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        # Create SI recording extractor object
        si_recording: si.BaseRecording = SI_READERS[acq_software](
            folder_path=output_dir
        )

        # Add probe information to recording object
        electrode_config_key = (
            probe.ElectrodeConfig * ephys.EphysRecording & key
        ).fetch1("KEY")
        electrodes_df = (
            (
                probe.ElectrodeConfig.Electrode * probe.ProbeType.Electrode
                & electrode_config_key
            )
            .fetch(format="frame")
            .reset_index()[["electrode", "x_coord", "y_coord", "shank"]]
        )

        # Create SI probe object
        si_probe = readers.probe_geometry.to_probeinterface(electrodes_df)
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        preprocessing_method = "catGT"  # where to load this info?
        si_recording = {
            "catGT": mimic_catGT,
            "IBLdestriping": mimic_IBLdestriping,
            "IBLdestriping_modified": mimic_IBLdestriping_modified,
        }[preprocessing_method](si_recording)
        recording_file = output_dir / "si_recording.pkl"
        si_recording.dump_to_pickle(file_path=recording_file)

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


@schema
class SIClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> PreProcessing
    sorter_name: varchar(30)        # name of the sorter used
    ---
    execution_time: datetime        # datetime of the start of this step
    execution_duration: float       # execution duration in hours
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load recording object.
        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_file = output_dir / "si_recording.pkl"
        si_recording: si.BaseRecording = si.load_extractor(recording_file)

        # Get sorter method and create output directory.
        clustering_method, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "params")
        sorter_name = (
            "kilosort_2_5" if clustering_method == "kilsort2.5" else clustering_method
        )
        sorter_dir = output_dir / sorter_name

        # Run sorting
        si_sorting: si.sorters.BaseSorter = si.sorters.run_sorter(
            sorter_name=sorter_name,
            recording=si_recording,
            output_folder=sorter_dir,
            verbse=True,
            docker_image=True,
            **params,
        )

        # Run sorting
        sorting_save_path = sorter_dir / "si_sorting.pkl"
        si_sorting.dump_to_pickle(sorting_save_path)

        self.insert1(
            {
                **key,
                "sorter_name": sorter_name,
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
