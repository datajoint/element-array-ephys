"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the
"ecephys_spike_sorting" pipeline.
The "ecephys_spike_sorting" was originally developed by the Allen Institute (https://github.com/AllenInstitute/ecephys_spike_sorting) for Neuropixels data acquired with Open Ephys acquisition system.
Then forked by Jennifer Colonell from the Janelia Research Campus (https://github.com/jenniferColonell/ecephys_spike_sorting) to support SpikeGLX acquisition system.

At DataJoint, we fork from Jennifer's fork and implemented a version that supports both Open Ephys and Spike GLX.
https://github.com/datajoint-company/ecephys_spike_sorting

The follow pipeline features intermediary tables:
1. KilosortPreProcessing - for preprocessing steps (no GPU required)
    - median_subtraction for Open Ephys
    - or the CatGT step for SpikeGLX
2. KilosortClustering - kilosort (MATLAB) - requires GPU
    - supports kilosort 2.0, 2.5 or 3.0 (https://github.com/MouseLand/Kilosort.git)
3. KilosortPostProcessing - for postprocessing steps (no GPU required)
    - kilosort_postprocessing
    - noise_templates
    - mean_waveforms
    - quality_metrics


"""
import datajoint as dj
import os
from element_array_ephys import get_logger
from decimal import Decimal
import json
import numpy as np
from datetime import datetime, timedelta

from element_interface.utils import find_full_path
from element_array_ephys.readers import (
    spikeglx,
    kilosort_triggering,
)
import element_array_ephys.ephys_no_curation as ephys
import element_array_ephys.probe as probe
# from element_array_ephys.ephys_no_curation import (
#     get_ephys_root_data_dir,
#     get_session_directory,
#     get_openephys_filepath,
#     get_spikeglx_meta_filepath,
#     get_recording_channels_details,
# )
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import spikeinterface.preprocessing as sip
import probeinterface as pi

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
class SI_preprocessing(dj.Imported):
    """A table to handle preprocessing of each clustering task."""

    definition = """
    -> ephys.ClusteringTask
    ---
    params: longblob           # finalized parameterset for this run
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    @property
    def key_source(self):
        return (
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & 'clustering_method in ("kilosort2", "kilosort2.5", "kilosort3")'
        ) - ephys.Clustering
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

        # add additional probe-recording and channels details into `params`
        params = {**params, **ephys.get_recording_channels_details(key)}
        params["fs"] = params["sample_rate"]

        if acq_software == "SpikeGLX":
            sglx_full_path = find_full_path(ephys.get_ephys_root_data_dir(),ephys.get_session_directory(key))
            sglx_filepath = ephys.get_spikeglx_meta_filepath(key)
            stream_name = os.path.split(sglx_filepath)[1]

            assert len(oe_probe.recording_info["recording_files"]) == 1

            # Create SI recording extractor object
            # sglx_si_recording = se.SpikeGLXRecordingExtractor(folder_path=sglx_full_path, stream_name=stream_name) 
            sglx_si_recording = se.read_spikeglx(folder_path=sglx_full_path, stream_name=stream_name) 
            electrode_query = (probe.ProbeType.Electrode
                * probe.ElectrodeConfig.Electrode
                * ephys.EphysRecording & key)

            xy_coords = [list(i) for i in zip(electrode_query.fetch('x_coord'),electrode_query.fetch('y_coord'))]
            channels_details = ephys.get_recording_channels_details(key)

            # Create SI probe object 
            probe = pi.Probe(ndim=2, si_units='um')
            probe.set_contacts(positions=xy_coords, shapes='square', shape_params={'width': 5})
            probe.create_auto_shape(probe_type='tip')
            channel_indices = np.arange(channels_details['num_channels'])
            probe.set_device_channel_indices(channel_indices)
            oe_si_recording.set_probe(probe=probe)

            # run preprocessing and save results to output folder
            sglx_si_recording_filtered = sip.bandpass_filter(sglx_si_recording, freq_min=300, freq_max=6000)
            sglx_recording_cmr = sip.common_reference(sglx_si_recording_filtered, reference="global", operator="median")
            sglx_recording_cmr.save_to_folder('sglx_recording_cmr', kilosort_dir)    


        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            oe_full_path = find_full_path(get_ephys_root_data_dir(),get_session_directory(key))
            oe_filepath = get_openephys_filepath(key)
            stream_name = os.path.split(oe_filepath)[1]

            assert len(oe_probe.recording_info["recording_files"]) == 1

            # Create SI recording extractor object
            # oe_si_recording = se.OpenEphysBinaryRecordingExtractor(folder_path=oe_full_path, stream_name=stream_name) 
            oe_si_recording = se.read_openephys(folder_path=oe_full_path, stream_name=stream_name) 
            electrode_query = (probe.ProbeType.Electrode
                * probe.ElectrodeConfig.Electrode
                * ephys.EphysRecording & key)

            xy_coords = [list(i) for i in zip(electrode_query.fetch('x_coord'),electrode_query.fetch('y_coord'))]
            channels_details = get_recording_channels_details(key)

            # Create SI probe object 
            probe = pi.Probe(ndim=2, si_units='um')
            probe.set_contacts(positions=xy_coords, shapes='square', shape_params={'width': 5})
            probe.create_auto_shape(probe_type='tip')
            channel_indices = np.arange(channels_details['num_channels'])
            probe.set_device_channel_indices(channel_indices)
            oe_si_recording.set_probe(probe=probe)

            # run preprocessing and save results to output folder
            oe_si_recording_filtered = sip.bandpass_filter(oe_si_recording, freq_min=300, freq_max=6000)
            oe_recording_cmr = sip.common_reference(oe_si_recording_filtered, reference="global", operator="median")
            oe_recording_cmr.save_to_folder('oe_recording_cmr', kilosort_dir)

        self.insert1(
            {
                **key,
                "params": params,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )           
@schema
class SI_KilosortClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> KilosortPreProcessing
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

        params = (KilosortPreProcessing & key).fetch1("params")

        if acq_software == "SpikeGLX":
            sglx_probe = ephys.get_openephys_probe_data(key)
            oe_si_recording = se.load_from_folder
            assert len(oe_probe.recording_info["recording_files"]) == 1
            if clustering_method.startswith('kilosort2.5'):
                sorter_name = "kilosort2_5"
            else:
                sorter_name = clustering_method
            sorting_kilosort = si.run_sorter(
                sorter_name = sorter_name,
                recording = oe_si_recording,
                output_folder = kilosort_dir,
                docker_image = f"spikeinterface/{sorter_name}-compiled-base:latest",
                **params
            )
            sorting_kilosort.save_to_folder('sorting_kilosort', kilosort_dir)
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            oe_si_recording = se.load_from_folder
            assert len(oe_probe.recording_info["recording_files"]) == 1
            if clustering_method.startswith('kilosort2.5'):
                sorter_name = "kilosort2_5"
            else:
                sorter_name = clustering_method
            sorting_kilosort = si.run_sorter(
                sorter_name = sorter_name,
                recording = oe_si_recording,
                output_folder = kilosort_dir,
                docker_image = f"spikeinterface/{sorter_name}-compiled-base:latest",
                **params
            )
            sorting_kilosort.save_to_folder('sorting_kilosort', kilosort_dir)

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
class KilosortPreProcessing(dj.Imported):
    """A processing table to handle each clustering task."""
    
    definition = """
    -> ephys.ClusteringTask
    ---
    params: longblob           # finalized parameterset for this run
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    @property
    def key_source(self):
        return (
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & 'clustering_method in ("kilosort2", "kilosort2.5", "kilosort3")'
        ) - ephys.Clustering

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

        # add additional probe-recording and channels details into `params`
        params = {**params, **ephys.get_recording_channels_details(key)}
        params["fs"] = params["sample_rate"]




        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            spikeglx_recording.validate_file("ap")
            run_CatGT = (
                params.get("run_CatGT", True)
                and "_tcat." not in spikeglx_meta_filepath.stem
            )

            run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                npx_input_dir=spikeglx_meta_filepath.parent,
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                run_CatGT=run_CatGT,
            )
            run_kilosort.run_CatGT()
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)

            assert len(oe_probe.recording_info["recording_files"]) == 1

            # run kilosort
            run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                npx_input_dir=oe_probe.recording_info["recording_files"][0],
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
            )
            run_kilosort._modules = ["depth_estimation", "median_subtraction"]
            run_kilosort.run_modules()

        self.insert1(
            {
                **key,
                "params": params,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )


@schema
class KilosortClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> KilosortPreProcessing
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

        params = (KilosortPreProcessing & key).fetch1("params")

        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            spikeglx_recording.validate_file("ap")

            run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                npx_input_dir=spikeglx_meta_filepath.parent,
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                run_CatGT=True,
            )
            run_kilosort._modules = ["kilosort_helper"]
            run_kilosort._CatGT_finished = True
            run_kilosort.run_modules()
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)

            assert len(oe_probe.recording_info["recording_files"]) == 1

            # run kilosort
            run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                npx_input_dir=oe_probe.recording_info["recording_files"][0],
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
            )
            run_kilosort._modules = ["kilosort_helper"]
            run_kilosort.run_modules()

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
class KilosortPostProcessing(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> KilosortClustering
    ---
    modules_status: longblob   # dictionary of summary status for all modules
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

        params = (KilosortPreProcessing & key).fetch1("params")

        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            spikeglx_recording.validate_file("ap")

            run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                npx_input_dir=spikeglx_meta_filepath.parent,
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                run_CatGT=True,
            )
            run_kilosort._modules = [
                "kilosort_postprocessing",
                "noise_templates",
                "mean_waveforms",
                "quality_metrics",
            ]
            run_kilosort._CatGT_finished = True
            run_kilosort.run_modules()
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)

            assert len(oe_probe.recording_info["recording_files"]) == 1

            # run kilosort
            run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                npx_input_dir=oe_probe.recording_info["recording_files"][0],
                ks_output_dir=kilosort_dir,
                params=params,
                KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
            )
            run_kilosort._modules = [
                "kilosort_postprocessing",
                "noise_templates",
                "mean_waveforms",
                "quality_metrics",
            ]
            run_kilosort.run_modules()

        with open(run_kilosort._modules_input_hash_fp) as f:
            modules_status = json.load(f)

        self.insert1(
            {
                **key,
                "modules_status": modules_status,
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
