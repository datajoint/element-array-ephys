import datajoint as dj
from element_array_ephys import get_logger
from decimal import Decimal
import json
from datetime import datetime, timedelta

from element_interface.utils import find_full_path
from element_array_ephys.readers import (
    spikeglx,
    kilosort,
    openephys,
    kilosort_triggering,
)

log = get_logger(__name__)

schema = dj.schema()

ephys = None


def activate(schema_name, ephys_schema_name, *, create_schema=True, create_tables=True):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `spike_sorting` schema
        :param ephys_schema_name: schema name of the activated ephys element for which this ephys_report schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    (The "activation" of this ephys_report module should be evoked by one of the ephys modules only)
    """
    global ephys
    ephys = dj.create_virtual_module("ephys", ephys_schema_name)
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=ephys.__dict__,
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
        )

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

        assert clustering_method in (
            "kilosort2",
            "kilosort2.5",
            "kilosort3",
        ), 'Supporting "kilosort" clustering_method only'

        # add additional probe-recording and channels details into `params`
        params = {**params, **ephys.get_recording_channels_details(key)}
        params["fs"] = params["sample_rate"]

        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            spikeglx_recording.validate_file("ap")
            run_CatGT = (
                params.pop("run_CatGT", True)
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
        assert clustering_method in (
            "kilosort2",
            "kilosort2.5",
            "kilosort3",
        ), 'Supporting "kilosort" clustering_method only'

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
        assert clustering_method in (
            "kilosort2",
            "kilosort2.5",
            "kilosort3",
        ), 'Supporting "kilosort" clustering_method only'

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

        with open(self._modules_input_hash_fp) as f:
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
