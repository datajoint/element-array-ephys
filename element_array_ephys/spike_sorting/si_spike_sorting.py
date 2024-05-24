"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the "spikeinterface" pipeline. Spikeinterface was developed by Alessio Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, and Matthias Hennig (https://github.com/SpikeInterface)
"""

from datetime import datetime

import datajoint as dj
import pandas as pd
import spikeinterface as si
from element_array_ephys import probe, readers
from element_interface.utils import find_full_path
from spikeinterface import exporters, postprocessing, qualitymetrics, sorters

from . import si_preprocessing

log = dj.logger

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
    ephys.Clustering.key_source -= PreProcessing.key_source.proj()


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

        # Set the output directory
        clustering_method, acq_software, output_dir, params = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "acq_software", "clustering_output_dir", "params")

        # Get sorter method and create output directory.
        sorter_name = clustering_method.replace(".", "_")

        for required_key in (
            "SI_SORTING_PARAMS",
            "SI_PREPROCESSING_METHOD",
            "SI_POSTPROCESSING_PARAMS",
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
            raise NotImplementedError(
                f"SpikeInterface processing for {acq_software} not yet implemented."
            )
        acq_software = acq_software.replace(" ", "").lower()
        si_extractor: si.extractors.neoextractors = (
            si.extractors.extractorlist.recording_extractor_full_dict[acq_software]
        )  # data extractor object

        stream_names, stream_ids = si.extractors.get_neo_streams(
            acq_software, folder_path=data_dir
        )
        si_recording: si.BaseRecording = si_extractor(
            folder_path=data_dir, stream_name=stream_names[0]
        )

        # Add probe information to recording object
        electrodes_df = (
            (
                ephys.EphysRecording.Channel
                * probe.ElectrodeConfig.Electrode
                * probe.ProbeType.Electrode
                & key
            )
            .fetch(format="frame")
            .reset_index()
        )

        # Create SI probe object
        si_probe = readers.probe_geometry.to_probeinterface(
            electrodes_df[["electrode", "x_coord", "y_coord", "shank"]]
        )
        si_probe.set_device_channel_indices(electrodes_df["channel_idx"].values)
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        si_preproc_func = getattr(si_preprocessing, params["SI_PREPROCESSING_METHOD"])
        si_recording = si_preproc_func(si_recording)
        si_recording.dump_to_pickle(file_path=recording_file, relative_to=output_dir)

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
        si_recording: si.BaseRecording = si.load_extractor(
            recording_file, base_folder=output_dir
        )

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

        # Save sorting object
        sorting_save_path = (
            output_dir / sorter_name / "spike_sorting" / "si_sorting.pkl"
        )
        si_sorting.dump_to_pickle(sorting_save_path, relative_to=output_dir)

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
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        recording_file = output_dir / sorter_name / "recording" / "si_recording.pkl"
        sorting_file = output_dir / sorter_name / "spike_sorting" / "si_sorting.pkl"

        si_recording: si.BaseRecording = si.load_extractor(
            recording_file, base_folder=output_dir
        )
        si_sorting: si.sorters.BaseSorter = si.load_extractor(
            sorting_file, base_folder=output_dir
        )

        # Sorting Analyzer
        analyzer_output_dir = output_dir / sorter_name / "sorting_analyzer"
        if (analyzer_output_dir / "extensions").exists():
            sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_output_dir)
        else:
            sorting_analyzer = si.create_sorting_analyzer(
                sorting=si_sorting,
                recording=si_recording,
                format="binary_folder",
                folder=analyzer_output_dir,
                sparse=True,
                overwrite=True,
            )

        job_kwargs = params["SI_POSTPROCESSING_PARAMS"].get("job_kwargs", {"n_jobs": -1, "chunk_duration": "1s"})
        extensions_params = params["SI_POSTPROCESSING_PARAMS"].get("extensions", {})
        # The order of extension computation is drawn from sorting_analyzer.get_computable_extensions()
        # each extension is parameterized by params specified in extensions_params dictionary (skip if not specified)
        extensions_to_compute = {ext_name: extensions_params[ext_name]
                                 for ext_name in sorting_analyzer.get_computable_extensions()
                                 if ext_name in extensions_params}

        sorting_analyzer.compute(extensions_to_compute, **job_kwargs)

        # Save to phy format
        if params["SI_POSTPROCESSING_PARAMS"].get("export_to_phy", False):
            si.exporters.export_to_phy(
                sorting_analyzer=sorting_analyzer, output_folder=output_dir / sorter_name / "phy",
                **job_kwargs
            )
        # Generate spike interface report
        if params["SI_POSTPROCESSING_PARAMS"].get("export_report", True):
            si.exporters.export_report(
                sorting_analyzer=sorting_analyzer,
                output_folder=output_dir / sorter_name / "spikeinterface_report",
                **job_kwargs
            )

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
