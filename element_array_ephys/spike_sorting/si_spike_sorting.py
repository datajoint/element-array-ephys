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

import pathlib
from datetime import datetime

import datajoint as dj
import pandas as pd
import probeinterface as pi
import spikeinterface as si
from element_array_ephys import get_logger, probe, readers
from element_interface.utils import find_full_path, memoized_result
from spikeinterface import exporters, postprocessing, qualitymetrics, sorters

from . import si_preprocessing

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
        clustering_method, acq_software, output_dir, params = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "acq_software", "clustering_output_dir", "params")
        
        # Get sorter method and create output directory.
        sorter_name = (
            "kilosort2_5" if clustering_method == "kilosort2.5" else clustering_method
        )
        
        for required_key in (
            "SI_SORTING_PARAMS",
            "SI_PREPROCESSING_METHOD",
            "SI_WAVEFORM_EXTRACTION_PARAMS",
            "SI_QUALITY_METRICS_PARAMS",
        ):
            if required_key not in params:
                raise ValueError(
                    f"{required_key} must be defined in ClusteringParamSet for SpikeInterface execution"
                )

        # Set directory to store recording file.
        if not output_dir:
            output_dir = ephys.ClusteringTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update clustering_output_dir
            ephys.ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_dir = output_dir / sorter_name / "recording"
        recording_dir.mkdir(parents=True, exist_ok=True)
        recording_file = (
            recording_dir / "si_recording.pkl"
        )  # recording cache to be created for each key

        # Create SI recording extractor object
        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = readers.spikeglx.SpikeGLX(
                spikeglx_meta_filepath.parent
            )
            spikeglx_recording.validate_file("ap")
            data_dir = spikeglx_meta_filepath.parent
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            assert len(oe_probe.recording_info["recording_files"]) == 1
            data_dir = oe_probe.recording_info["recording_files"][0]
        else:
            raise NotImplementedError(f"Not implemented for {acq_software}")

        stream_names, stream_ids = si.extractors.get_neo_streams(
            acq_software.strip().lower(), folder_path=data_dir
        )
        si_recording: si.BaseRecording = SI_READERS[acq_software](
            folder_path=data_dir, stream_name=stream_names[0]
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
        si_probe.set_device_channel_indices(range(len(electrodes_df)))
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        si_preproc_func = si_preprocessing.preprocessing_function_mapping[
            params["SI_PREPROCESSING_METHOD"]
        ]
        si_recording = si_preproc_func(si_recording)
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
    ---
    execution_time: datetime        # datetime of the start of this step
    execution_duration: float       # execution duration in hours
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load recording object.
        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_file = output_dir / "si_recording.pkl"
        si_recording: si.BaseRecording = si.load_extractor(recording_file)

        # Get sorter method and create output directory.
        sorter_name = (
            "kilosort2_5" if clustering_method == "kilosort2.5" else clustering_method
        )

        # Run sorting
        @memoized_result(
            parameters={**key, **params},
            output_directory=output_dir / sorter_name,
        )
        def _run_sorter(*args, **kwargs):
            si_sorting: si.sorters.BaseSorter = si.sorters.run_sorter(*args, **kwargs)
            sorting_save_path = output_dir / sorter_name / "si_sorting.pkl"
            si_sorting.dump_to_pickle(sorting_save_path)
            return sorting_save_path

        sorting_save_path = _run_sorter(
            sorter_name=sorter_name,
            recording=si_recording,
            output_folder=output_dir / sorter_name,
            remove_existing_folder=True,
            verbose=True,
            docker_image=True,
            **params.get("SI_SORTING_PARAMS", {}),
        )

        # Run sorting
        sorting_save_path = output_dir / "si_sorting.pkl"
        si_sorting.dump_to_pickle(sorting_save_path)

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
class PostProcessing(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> SIClustering
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load sorting & recording object.
        output_dir, params = (ephys.ClusteringTask & key).fetch1(
            "clustering_output_dir", "params"
        )
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_file = output_dir / "si_recording.pkl"
        sorting_file = output_dir / "si_sorting.pkl"

        si_recording: si.BaseRecording = si.load_extractor(recording_file)
        si_sorting: si.sorters.BaseSorter = si.load_extractor(sorting_file)

        # Extract waveforms
        we: si.WaveformExtractor = si.extract_waveforms(
            si_recording,
            si_sorting,
            folder=output_dir / "waveform",  # The folder where waveforms are cached
            max_spikes_per_unit=None,
            overwrite=True,
            **params.get("SI_WAVEFORM_EXTRACTION_PARAMS", {}),
            **params.get("SI_JOB_KWARGS", {"n_jobs": -1, "chunk_size": 30000}),
        )

        # Calculate QC Metrics
        metrics: pd.DataFrame = si.qualitymetrics.compute_quality_metrics(
            we,
            metric_names=[
                "firing_rate",
                "snr",
                "presence_ratio",
                "isi_violation",
                "num_spikes",
                "amplitude_cutoff",
                "amplitude_median",
                "sliding_rp_violation",
                "rp_violation",
                "drift",
            ],
        )
        # Add PCA based metrics. These will be added to the metrics dataframe above.
        _ = si.postprocessing.compute_principal_components(
            waveform_extractor=we, **params.get("SI_QUALITY_METRICS_PARAMS", None)
        )
        # Save the output (metrics.csv to the output dir)
        metrics = si.qualitymetrics.compute_quality_metrics(waveform_extractor=we)
        metrics.to_csv(output_dir / "metrics.csv")

        # Save results
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

        # Once finished, insert this `key` into ephys.Clustering
        ephys.Clustering.insert1(
            {**key, "clustering_time": datetime.utcnow()}, allow_direct_insert=True
        )
