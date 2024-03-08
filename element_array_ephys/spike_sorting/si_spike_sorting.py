"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the "spikeinterface" pipeline.
Spikeinterface was developed by Alessio Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, and Matthias Hennig (https://github.com/SpikeInterface)
"""

from datetime import datetime

import datajoint as dj
import numpy as np
import pandas as pd
import spikeinterface as si
from element_interface.utils import find_full_path
from spikeinterface import exporters, postprocessing, qualitymetrics, sorters

from .. import get_logger, probe, readers
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

        # Get clustering method and output directory.
        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        acq_software = (ephys.EphysRawFile & key).fetch("acq_software", limit=1)[0]

        # Get sorter method and create output directory.
        sorter_name = clustering_method.replace(".", "_")

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
        recording_file = recording_dir / "si_recording.pkl"

        # Get probe information to recording object
        probe_info = (ephys.EphysSessionProbe & key).fetch1()
        probe_type = ((probe.Probe * ephys.EphysSessionProbe()) & key).fetch1(
            "probe_type"
        )

        electrode_query = probe.ElectrodeConfig.Electrode & (
            probe.ElectrodeConfig & {"probe_type": probe_type}
        )

        # Filter for used electrodes. If probe_info["used_electrodes"] is None, it means all electrodes were used.
        number_of_electrodes = len(electrode_query)
        probe_info["used_electrodes"] = probe_info["used_electrodes"] or list(
            range(number_of_electrodes)
        )
        unused_electrodes = [
            elec
            for elec in range(number_of_electrodes)
            if elec not in probe_info["used_electrodes"]
        ]
        electrode_query &= f'electrode IN {tuple(probe_info["used_electrodes"])}'
        electrodes_df = (
            (probe.ProbeType.Electrode * electrode_query)
            .fetch(format="frame", order_by="electrode")
            .reset_index()[["electrode", "x_coord", "y_coord", "shank", "channel"]]
        )

        def get_port_indices(file_path, port_id):
            """Get the row indices of the port from the data matrix."""
            import intanrhdreader

            data = intanrhdreader.load_file(file_path)
            return np.array(
                [
                    ind
                    for ind, ch in enumerate(data["amplifier_channels"])
                    if ch["port_prefix"] == port_id
                ]
            )

        # Create SI recording extractor object
        si_extractor: si.extractors.neoextractors = (
            si.extractors.extractorlist.recording_extractor_full_dict[
                acq_software.replace(" ", "").lower()
            ]
        )  # data extractor object

        files, file_times = (
            ephys.EphysRawFile
            & key
            & f"file_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
        ).fetch("file_path", "file_time", order_by="file_time")

        si_recording = None

        # Read data. Concatenate if multiple files are found.
        for file_path in (
            find_full_path(ephys.get_ephys_root_data_dir(), f) for f in files
        ):
            if not si_recording:
                stream_name = [
                    s
                    for s in si_extractor.get_streams(file_path)[0]
                    if "amplifier" in s
                ][0]
                si_recording: si.BaseRecording = si_extractor(
                    file_path, stream_name=stream_name
                )
                port_indices = get_port_indices(
                    file_path, port_id=probe_info["port_id"]
                )  # get the row indices of the port
            else:
                si_recording: si.BaseRecording = si.concatenate_recordings(
                    [
                        si_recording,
                        si_extractor(file_path, stream_name=stream_name),
                    ]
                )

        si_recording = si_recording.channel_slice(
            si_recording.channel_ids[port_indices]
        )  # select only the port data

        # Create SI probe object
        si_probe = readers.probe_geometry.to_probeinterface(electrodes_df)
        si_probe.set_device_channel_indices(electrodes_df.channel.values)
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        if unused_electrodes:
            electrode_to_index_map = dict(
                zip(electrodes_df["electrode"], electrodes_df["channel"])
            )  # electrode to channel index (data row index)
            channel_index_to_remove = np.array(
                sorted(
                    [
                        int(electrode_to_index_map[e]) + port_indices.min()
                        for e in unused_electrodes
                    ]
                )
            )
            channel_index_to_remove = list(map(str, channel_index_to_remove))
            si_recording = si_recording.remove_channels(
                remove_channel_ids=channel_index_to_remove
            )

        si_preproc_func = getattr(si_preprocessing, params["SI_PREPROCESSING_METHOD"])
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
        sorter_name = clustering_method.replace(".", "_")
        recording_file = output_dir / sorter_name / "recording" / "si_recording.pkl"
        si_recording: si.BaseRecording = si.load_extractor(recording_file)

        # Run sorting
        # Sorting performed in a dedicated docker environment if the sorter is not built in the spikeinterface package.
        si_sorting: si.sorters.BaseSorter = si.sorters.run_sorter(
            sorter_name=sorter_name,
            recording=si_recording,
            output_folder=output_dir / sorter_name / "spike_sorting",
            remove_existing_folder=True,
            verbose=True,
            docker_image=sorter_name not in si.sorters.installed_sorters(),
            **params.get("SI_SORTING_PARAMS", {}),
        )

        # Save sorting object.
        sorting_save_path = (
            output_dir / sorter_name / "spike_sorting" / "si_sorting.pkl"
        )
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

        # Load recording & sorting object.
        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")
        recording_file = output_dir / sorter_name / "recording" / "si_recording.pkl"
        sorting_file = output_dir / sorter_name / "spike_sorting" / "si_sorting.pkl"

        si_recording: si.BaseRecording = si.load_extractor(recording_file)
        si_sorting: si.sorters.BaseSorter = si.load_extractor(sorting_file)

        # Extract waveforms
        we: si.WaveformExtractor = si.extract_waveforms(
            si_recording,
            si_sorting,
            folder=output_dir
            / sorter_name
            / "waveform",  # The folder where waveforms are cached
            overwrite=True,
            allow_unfiltered=True,
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
        # Save metrics.csv to the output dir
        metrics_output_dir = output_dir / sorter_name / "metrics"
        metrics_output_dir.mkdir(parents=True, exist_ok=True)

        metrics = si.qualitymetrics.compute_quality_metrics(waveform_extractor=we)
        metrics.to_csv(metrics_output_dir / "metrics.csv")

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
