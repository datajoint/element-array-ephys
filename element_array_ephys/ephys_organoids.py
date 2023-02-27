import gc
import importlib
import inspect
import pathlib
import re
from datetime import datetime, timedelta
from decimal import Decimal

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory
from intanrhsreader import load_file

from . import ephys_report, get_logger, probe
from .readers import kilosort, openephys, spikeglx

log = get_logger(__name__)

schema = dj.schema()

_linking_module = None


def activate(
    ephys_schema_name: str,
    probe_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activates the `ephys` and `probe` schemas.

    Args:
        ephys_schema_name (str): A string containing the name of the ephys schema.
        probe_schema_name (str): A string containing the name of the probe schema.
        create_schema (bool): If True, schema will be created in the database.
        create_tables (bool): If True, tables related to the schema will be created in the database.
        linking_module (str): A string containing the module name or module containing the required dependencies to activate the schema.

    Dependencies:
    Upstream tables:
        Probe: A parent table to EphysSession. Probe information is required before electrophysiology data is imported.

    Functions:
        get_ephys_root_data_dir(): Returns absolute path for root data director(y/ies) with all electrophysiological recording sessions, as a list of string(s).
        get_session_direction(session_key: dict): Returns path to electrophysiology data for the a particular session as a list of strings.
        get_processed_data_dir(): Optional. Returns absolute path for processed data. Defaults to root directory.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    probe.activate(
        probe_schema_name, create_schema=create_schema, create_tables=create_tables
    )
    schema.activate(
        ephys_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )
    ephys_report.activate(f"{ephys_schema_name}_report", ephys_schema_name)


# -------------- Functions required by the elements-ephys  ---------------


def get_ephys_root_data_dir() -> list:
    """Fetches absolute data path to ephys data directories.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path(s) to ephys data directories.
    """
    root_directories = _linking_module.get_ephys_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_session_directory(session_key: dict) -> str:
    """Retrieve the session directory with Neuropixels for the given session.

    Args:
        session_key (dict): A dictionary mapping subject to an entry in the subject table, and session_datetime corresponding to a session in the database.

    Returns:
        A string for the path to the session directory.
    """
    return _linking_module.get_session_directory(session_key)


def get_processed_root_data_dir() -> str:
    """Retrieve the root directory for all processed data.

    Returns:
        A string for the full path to the root directory for processed data.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_ephys_root_data_dir()[0]


# ----------------------------- Table declarations ----------------------
@schema
class AcquisitionSoftware(dj.Lookup):
    """Name of software used for recording electrophysiological data.

    Attributes:
        acq_software ( varchar(24) ): Acquisition software, e.g,. SpikeGLX, OpenEphys
    """

    definition = """  # Software used for recording of neuropixels probes
    acq_software: varchar(24)
    """
    contents = zip(["SpikeGLX", "Open Ephys", "Intan"])


@schema
class EphysSession(dj.Manual):
    """User defined ephys session for downstream analysis."""

    definition = """
    -> Subject
    -> probe.ElectrodeConfig # probe electrode config from probe.yaml
    start_time                  : datetime
    end_time                    : datetime
    ---
    session_type                : enum("lfp", "spike_sorting", "both")  # analysis target
    """


@schema
class RawData(dj.Imported):
    definition = """
    -> Subject
    start_time      : datetime(6) # date and time of file creation
    ---
    end_time        : datetime(6) # date and time of the end of the file
    file_name       : varchar(32) # name of the file
    file_path       : filepath@external-raw  
    sampling_rate   : float # (Hz)
    timestamps      : longblob # (s) timestamps with respect to the start of the recording (recording_timestamp)
    header          : longblob  # meta information about the file.
    """

    class Attributes(dj.Part):
        definition = """
        # All non-data attributes read from raw data file.
        -> master
        attribute_name      : varchar(32)
        ---
        attribute_blob=null : longblob
        """

    class Trace(dj.Part):
        definition = """
        # Raw ephys trace per channel.
        -> master
        channel_id  : varchar(16) # channel id read from the raw file.
        ---
        trace       : blob@external-raw # raw ephys trace from this channel in microvolts.
        """

    def make(self, key):
        data_dir = get_ephys_root_data_dir()[0]
        data_files = data_dir.glob(f"{key['induction_id']}*.rhs")

        recording_start_time = 0  # start of the acquisition.

        # Loop through all the data files.
        for file in sorted(list(data_files)):
            # Load data
            data = load_file(file)
            timestamps = data.pop("t")  # timestamp doesn't reset to 0 for each file.
            start_time = re.search(r".*_(\d{6}_\d{6})", file.stem).groups()[0]
            start_time = np.datetime64(
                datetime.strptime(start_time, "%y%m%d_%H%M%S")
            )  # start time based on the file name

            if not recording_start_time:
                recording_start_time = start_time - (
                    timestamps[0] * 10**6 * np.timedelta64(1, "us")
                )  # get the start time of the acquisition.
            timestamps = recording_start_time + (
                timestamps * 10**6 * np.timedelta64(1, "us")
            )

            self.insert1(
                {
                    **key,
                    "start_time": timestamps[0],
                    "end_time": timestamps[-1],
                    "file_name": file.relative_to(data_dir),
                    "file_path": file,
                    "sampling_rate": data["header"]["sample_rate"],
                    "timestamps": timestamps,
                    "header": data.pop("header"),
                }
            )

            attribute_list = [
                {
                    **key,
                    "start_time": timestamps[0],
                    "attribute_name": k,
                    "attribute_blob": v,
                }
                for k, v in data.items()
                if "data" not in k
            ]  # don't insert data here
            self.Attributes.insert(attribute_list)

            trace_channels = [
                ch["native_channel_name"] for ch in data["amplifier_channels"]
            ]  # channels with raw ephys traces

            trace_list = [
                {
                    **key,
                    "start_time": timestamps[0],
                    "channel_id": ch,
                    "trace": trace,
                }
                for ch, trace in zip(trace_channels, data.pop("amplifier_data"))
            ]
            self.Trace.insert(trace_list)


@schema
class LFP(dj.Imported):
    definition = """
    -> EphysSession
    ---
    lfp_sampling_rate       : float # (Hz) down-sampled sampling rate.
    lfp_time_stamps         : blob@external-raw # (s) 
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode
        ---
        lfp                 : blob@external-raw 
        """

    @property
    def key_source(self):
        return EphysSession - "session_type='spike_sorting'"

    def make(self, key):
        DS_FACTOR = 10  # downsampling factor

        timestamp_concat = np.array([], dtype=np.datetime64)  # initialize array

        query = (
            RawData
            & f"start_time >= '{key['start_time']}'"
            & f"end_time <= '{key['end_time']}'"
        )

        header = query.fetch("header")[0]
        lfp_sampling_rate = header["sample_rate"] / DS_FACTOR

        # Downsample & concatenate timestamps.
        for timestamps in query.fetch("timestamps"):
            timestamp_concat = np.concatenate(
                (timestamp_concat, timestamps[::DS_FACTOR]), axis=0
            )

        self.insert1(
            {
                **key,
                "lfp_sampling_rate": lfp_sampling_rate,
                "lfp_time_stamps": timestamp_concat,
            }
        )

        channels = (probe.ElectrodeConfig.Electrode & key).fetch("channel_id")

        lfp_query = RawData.Trace * (probe.ElectrodeConfig.Electrode & key) & query

        # Single insert in loop to mitigate potential memory issue.
        for ch in channels:
            q = lfp_query & f"channel_id='{ch}'"
            probe_type, electrode, channel_id, lfps = q.fetch(
                "probe_type", "electrode", "channel_id", "trace", order_by="start_time"
            )

            lfp_concat = np.array([], dtype=np.float32)  # initialize array

            # Downsample & concatenate LFP traces.
            for lfp in lfps:
                lfp_concat = np.concatenate((lfp_concat, lfp[::DS_FACTOR]), axis=0)

            self.Electrode.insert1(
                {
                    **key,
                    "probe_type": probe_type[0],
                    "electrode": electrode[0],
                    "channel_id": channel_id[0],
                    "lfp": lfp_concat,
                }
            )


# ------------ Clustering --------------


@schema
class ClusteringMethod(dj.Lookup):
    """Kilosort clustering method.

    Attributes:
        clustering_method (foreign key, varchar(16) ): Kilosort clustering method.
        clustering_methods_desc (varchar(1000) ): Additional description of the clustering method.
    """

    definition = """
    # Method for clustering
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [
        ("kilosort2", "kilosort2 clustering method"),
        ("kilosort2.5", "kilosort2.5 clustering method"),
        ("kilosort3", "kilosort3 clustering method"),
    ]


@schema
class ClusteringParamSet(dj.Lookup):
    """Parameters to be used in clustering procedure for spike sorting.

    Attributes:
        paramset_idx (foreign key): Unique ID for the clustering parameter set.
        ClusteringMethod (dict): ClusteringMethod primary key.
        paramset_desc (varchar(128) ): Description of the clustering parameter set.
        param_set_hash (uuid): UUID hash for the parameter set.
        params (longblob): Parameters for clustering with Kilosort.
    """

    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(
        cls,
        clustering_method: str,
        paramset_desc: str,
        params: dict,
        paramset_idx: int = None,
    ):
        """Inserts new parameters into the ClusteringParamSet table.

        Args:
            clustering_method (str): name of the clustering method.
            paramset_desc (str): description of the parameter set
            params (dict): clustering parameters
            paramset_idx (int, optional): Unique parameter set ID. Defaults to None.
        """
        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "clustering_method": clustering_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(
                {**params, "clustering_method": clustering_method}
            ),
        }
        param_query = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if (
                existing_paramset_idx == paramset_idx
            ):  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f"The specified param-set already exists"
                    f" - with paramset_idx: {existing_paramset_idx}"
                )
        else:
            if {"paramset_idx": paramset_idx} in cls.proj():
                raise dj.DataJointError(
                    f"The specified paramset_idx {paramset_idx} already exists,"
                    f" please pick a different one."
                )
            cls.insert1(param_dict)


@schema
class ClusterQualityLabel(dj.Lookup):
    """Quality label for each spike sorted cluster.

    Attributes:
        cluster_quality_label (foreign key, varchar(100) ): Cluster quality type.
        cluster_quality_description (varchar(4000) ): Description of the cluster quality type.
    """

    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ("good", "single unit"),
        ("ok", "probably a single unit, but could be contaminated"),
        ("mua", "multi-unit activity"),
        ("noise", "bad unit"),
    ]


@schema
class ClusteringTask(dj.Manual):
    """A clustering task to spike sort electrophysiology datasets.

    Attributes:
        EphysSession (foreign key): EphysSession primary key.
        ClusteringParamSet (foreign key): ClusteringParamSet primary key.
        clustering_outdir_dir (varchar (255) ): Relative path to output clustering results.
        task_mode (enum): `Trigger` computes clustering or and `load` imports existing data.
    """

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysSession
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @property
    def key_source(self):
        return EphysSession - "session_type='lfp'"

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False) -> pathlib.Path:
        """Infer output directory if it is not provided.

        Args:
            key (dict): ClusteringTask primary key.

        Returns:
            Expected clustering_output_dir based on the following convention:
                processed_dir / session_dir / probe_{insertion_number} / {clustering_method}_{paramset_idx}
                e.g.: sub4/sess1/probe_2/kilosort2_0
        """
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        sess_dir = find_full_path(get_ephys_root_data_dir(), get_session_directory(key))
        root_dir = find_root_directory(get_ephys_root_data_dir(), sess_dir)

        method = (
            (ClusteringParamSet * ClusteringMethod & key)
            .fetch1("clustering_method")
            .replace(".", "-")
        )

        output_dir = (
            processed_dir
            / sess_dir.relative_to(root_dir)
            / f'probe_{key["insertion_number"]}'
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"{output_dir} created!")

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, ephys_recording_key: dict, paramset_idx: int = 0):
        """Autogenerate entries based on a particular ephys recording.

        Args:
            ephys_recording_key (dict): EphysSession primary key.
            paramset_idx (int, optional): Parameter index to use for clustering task. Defaults to 0.
        """
        key = {**ephys_recording_key, "paramset_idx": paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = ClusteringTask.infer_output_dir(key, relative=False, mkdir=True)

        try:
            kilosort.Kilosort(
                output_dir
            )  # check if the directory is a valid Kilosort output
        except FileNotFoundError:
            task_mode = "trigger"
        else:
            task_mode = "load"

        cls.insert1(
            {
                **key,
                "clustering_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
                "task_mode": task_mode,
            }
        )


@schema
class Clustering(dj.Imported):
    """A processing table to handle each clustering task.

    Attributes:
        ClusteringTask (foreign key): ClusteringTask primary key.
        clustering_time (datetime): Time when clustering results are generated.
        package_version (varchar(16) ): Package version used for a clustering analysis.
    """

    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results
    package_version='': varchar(16)
    """

    def make(self, key):
        """Triggers or imports clustering analysis."""
        task_mode, output_dir = (ClusteringTask & key).fetch1(
            "task_mode", "clustering_output_dir"
        )

        if not output_dir:
            output_dir = ClusteringTask.infer_output_dir(key, relative=True, mkdir=True)
            # update clustering_output_dir
            ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )

        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == "load":
            kilosort.Kilosort(
                kilosort_dir
            )  # check if the directory is a valid Kilosort output
        elif task_mode == "trigger":
            acq_software, clustering_method, params = (
                ClusteringTask * EphysSession * ClusteringParamSet & key
            ).fetch1("acq_software", "clustering_method", "params")

            if "kilosort" in clustering_method:
                from element_array_ephys.readers import kilosort_triggering

                # add additional probe-recording and channels details into `params`
                params = {**params, **get_recording_channels_details(key)}
                params["fs"] = params["sample_rate"]

                if acq_software == "SpikeGLX":
                    spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                    spikeglx_recording = spikeglx.SpikeGLX(
                        spikeglx_meta_filepath.parent
                    )
                    spikeglx_recording.validate_file("ap")
                    run_CatGT = (
                        params.pop("run_CatGT", True)
                        and "_tcat." not in spikeglx_meta_filepath.stem
                    )

                    if clustering_method.startswith("pykilosort"):
                        kilosort_triggering.run_pykilosort(
                            continuous_file=spikeglx_recording.root_dir
                            / (spikeglx_recording.root_name + ".ap.bin"),
                            kilosort_output_directory=kilosort_dir,
                            channel_ind=params.pop("channel_ind"),
                            x_coords=params.pop("x_coords"),
                            y_coords=params.pop("y_coords"),
                            shank_ind=params.pop("shank_ind"),
                            connected=params.pop("connected"),
                            sample_rate=params.pop("sample_rate"),
                            params=params,
                        )
                    else:
                        run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                            npx_input_dir=spikeglx_meta_filepath.parent,
                            ks_output_dir=kilosort_dir,
                            params=params,
                            KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                            run_CatGT=run_CatGT,
                        )
                        run_kilosort.run_modules()
                elif acq_software == "Open Ephys":
                    oe_probe = get_openephys_probe_data(key)

                    assert len(oe_probe.recording_info["recording_files"]) == 1

                    # run kilosort
                    if clustering_method.startswith("pykilosort"):
                        kilosort_triggering.run_pykilosort(
                            continuous_file=pathlib.Path(
                                oe_probe.recording_info["recording_files"][0]
                            )
                            / "continuous.dat",
                            kilosort_output_directory=kilosort_dir,
                            channel_ind=params.pop("channel_ind"),
                            x_coords=params.pop("x_coords"),
                            y_coords=params.pop("y_coords"),
                            shank_ind=params.pop("shank_ind"),
                            connected=params.pop("connected"),
                            sample_rate=params.pop("sample_rate"),
                            params=params,
                        )
                    else:
                        run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                            npx_input_dir=oe_probe.recording_info["recording_files"][0],
                            ks_output_dir=kilosort_dir,
                            params=params,
                            KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                        )
                        run_kilosort.run_modules()
            else:
                raise NotImplementedError(
                    f"Automatic triggering of {clustering_method}"
                    f" clustering analysis is not yet supported"
                )

        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        creation_time, _, _ = kilosort.extract_clustering_info(kilosort_dir)
        self.insert1({**key, "clustering_time": creation_time})


@schema
class Curation(dj.Manual):
    """Curation procedure table.

    Attributes:
        Clustering (foreign key): Clustering primary key.
        curation_id (foreign key, int): Unique curation ID.
        curation_time (datetime): Time when curation results are generated.
        curation_output_dir (varchar(255) ): Output directory of the curated results.
        quality_control (bool): If True, this clustering result has undergone quality control.
        manual_curation (bool): If True, manual curation has been performed on this clustering result.
        curation_note (varchar(2000) ): Notes about the curation task.
    """

    definition = """
    # Manual curation procedure
    -> Clustering
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated clustering results
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    quality_control: bool               # has this clustering result undergone quality control?
    manual_curation: bool               # has manual curation been performed on this clustering result?
    curation_note='': varchar(2000)
    """

    def create1_from_clustering_task(self, key, curation_note: str = ""):
        """
        A function to create a new corresponding "Curation" for a particular
        "ClusteringTask"
        """
        if key not in Clustering():
            raise ValueError(
                f"No corresponding entry in Clustering available"
                f" for: {key}; do `Clustering.populate(key)`"
            )

        task_mode, output_dir = (ClusteringTask & key).fetch1(
            "task_mode", "clustering_output_dir"
        )
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        creation_time, is_curated, is_qc = kilosort.extract_clustering_info(
            kilosort_dir
        )
        # Synthesize curation_id
        curation_id = (
            dj.U().aggr(self & key, n="ifnull(max(curation_id)+1,1)").fetch1("n")
        )
        self.insert1(
            {
                **key,
                "curation_id": curation_id,
                "curation_time": creation_time,
                "curation_output_dir": output_dir,
                "quality_control": is_qc,
                "manual_curation": is_curated,
                "curation_note": curation_note,
            }
        )


@schema
class CuratedClustering(dj.Imported):
    """Clustering results after curation.

    Attributes:
        Curation (foreign key): Curation primary key.
    """

    definition = """
    # Clustering results of a curation.
    -> Curation
    """

    class Unit(dj.Part):
        """Single unit properties after clustering and curation.

        Attributes:
            CuratedClustering (foreign key): CuratedClustering primary key.
            unit (foreign key, int): Unique integer identifying a single unit.
            probe.ElectrodeConfig.Electrode (dict): probe.ElectrodeConfig.Electrode primary key.
            ClusteringQualityLabel (dict): CLusteringQualityLabel primary key.
            spike_count (int): Number of spikes in this recording for this unit.
            spike_times (longblob): Spike times of this unit, relative to start time of EphysSession.
            spike_sites (longblob): Array of electrode associated with each spike.
            spike_depths (longblob): Array of depths associated with each spike, relative to each spike.
        """

        definition = """
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording for this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysSession
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe
        """

    def make(self, key):
        """Automated population of Unit information."""
        output_dir = (Curation & key).fetch1("curation_output_dir")
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)
        acq_software, sample_rate = (EphysSession & key).fetch1(
            "acq_software", "sampling_rate"
        )

        sample_rate = kilosort_dataset.data["params"].get("sample_rate", sample_rate)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [
            i
            for i, u in enumerate(kilosort_dataset.data["cluster_ids"])
            if (kilosort_dataset.data["spike_clusters"] == u).any()
        ]
        valid_units = kilosort_dataset.data["cluster_ids"][withspike_idx]
        valid_unit_labels = kilosort_dataset.data["cluster_groups"][withspike_idx]
        # -- Get channel and electrode-site mapping
        channel2electrodes = get_neuropixels_channel2electrode_map(key, acq_software)

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = (
            "spike_times_sec_adj"
            if "spike_times_sec_adj" in kilosort_dataset.data
            else "spike_times_sec"
            if "spike_times_sec" in kilosort_dataset.data
            else "spike_times"
        )
        spike_times = kilosort_dataset.data[spike_time_key]
        kilosort_dataset.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = np.array(
            [
                channel2electrodes[s]["electrode"]
                for s in kilosort_dataset.data["spike_sites"]
            ]
        )
        spike_depths = kilosort_dataset.data["spike_depths"]

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data["spike_clusters"] == unit).any():
                unit_channel, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (
                    spike_times[kilosort_dataset.data["spike_clusters"] == unit]
                    / sample_rate
                )
                spike_count = len(unit_spike_times)

                units.append(
                    {
                        "unit": unit,
                        "cluster_quality_label": unit_lbl,
                        **channel2electrodes[unit_channel],
                        "spike_times": unit_spike_times,
                        "spike_count": spike_count,
                        "spike_sites": spike_sites[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ],
                        "spike_depths": spike_depths[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ]
                        if spike_depths is not None
                        else None,
                    }
                )

        self.insert1(key)
        self.Unit.insert([{**key, **u} for u in units])


@schema
class WaveformSet(dj.Imported):
    """A set of spike waveforms for units out of a given CuratedClustering.

    Attributes:
        CuratedClustering (foreign key): CuratedClustering primary key.
    """

    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        """Mean waveform across spikes for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            peak_electrode_waveform (longblob): Mean waveform for a given unit at its representative electrode.
        """

        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        """Spike waveforms for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            waveform_mean (longblob): mean waveform across spikes of the unit in microvolts.
            waveforms (longblob): waveforms of a sampling of spikes at the given electrode and unit.
        """

        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode
        ---
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """

    def make(self, key):
        """Populates waveform tables."""
        output_dir = (Curation & key).fetch1("curation_output_dir")
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)

        acq_software, probe_serial_number = (
            EphysSession * ProbeInsertion & key
        ).fetch1("acq_software", "probe")

        # -- Get channel and electrode-site mapping
        recording_key = (EphysSession & key).fetch1("KEY")
        channel2electrodes = get_neuropixels_channel2electrode_map(
            recording_key, acq_software
        )

        is_qc = (Curation & key).fetch1("quality_control")

        # Get all units
        units = {
            u["unit"]: u
            for u in (CuratedClustering.Unit & key).fetch(as_dict=True, order_by="unit")
        }

        if is_qc:
            unit_waveforms = np.load(
                kilosort_dir / "mean_waveforms.npy"
            )  # unit x channel x sample

            def yield_unit_waveforms():
                for unit_no, unit_waveform in zip(
                    kilosort_dataset.data["cluster_ids"], unit_waveforms
                ):
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []
                    if unit_no in units:
                        for channel, channel_waveform in zip(
                            kilosort_dataset.data["channel_map"], unit_waveform
                        ):
                            unit_electrode_waveforms.append(
                                {
                                    **units[unit_no],
                                    **channel2electrodes[channel],
                                    "waveform_mean": channel_waveform,
                                }
                            )
                            if (
                                channel2electrodes[channel]["electrode"]
                                == units[unit_no]["electrode"]
                            ):
                                unit_peak_waveform = {
                                    **units[unit_no],
                                    "peak_electrode_waveform": channel_waveform,
                                }
                    yield unit_peak_waveform, unit_electrode_waveforms

        else:
            if acq_software == "SpikeGLX":
                spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                neuropixels_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            elif acq_software == "Open Ephys":
                session_dir = find_full_path(
                    get_ephys_root_data_dir(), get_session_directory(key)
                )
                openephys_dataset = openephys.OpenEphys(session_dir)
                neuropixels_recording = openephys_dataset.probes[probe_serial_number]

            def yield_unit_waveforms():
                for unit_dict in units.values():
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []

                    spikes = unit_dict["spike_times"]
                    waveforms = neuropixels_recording.extract_spike_waveforms(
                        spikes, kilosort_dataset.data["channel_map"]
                    )  # (sample x channel x spike)
                    waveforms = waveforms.transpose(
                        (1, 2, 0)
                    )  # (channel x spike x sample)
                    for channel, channel_waveform in zip(
                        kilosort_dataset.data["channel_map"], waveforms
                    ):
                        unit_electrode_waveforms.append(
                            {
                                **unit_dict,
                                **channel2electrodes[channel],
                                "waveform_mean": channel_waveform.mean(axis=0),
                                "waveforms": channel_waveform,
                            }
                        )
                        if (
                            channel2electrodes[channel]["electrode"]
                            == unit_dict["electrode"]
                        ):
                            unit_peak_waveform = {
                                **unit_dict,
                                "peak_electrode_waveform": channel_waveform.mean(
                                    axis=0
                                ),
                            }

                    yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue
        self.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)


@schema
class QualityMetrics(dj.Imported):
    """Clustering and waveform quality metrics.

    Attributes:
        CuratedClustering (foreign key): CuratedClustering primary key.
    """

    definition = """
    # Clusters and waveforms metrics
    -> CuratedClustering
    """

    class Cluster(dj.Part):
        """Cluster metrics for a unit.

        Attributes:
            QualityMetrics (foreign key): QualityMetrics primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            firing_rate (float): Firing rate of the unit.
            snr (float): Signal-to-noise ratio for a unit.
            presence_ratio (float): Fraction of time where spikes are present.
            isi_violation (float): rate of ISI violation as a fraction of overall rate.
            number_violation (int): Total ISI violations.
            amplitude_cutoff (float): Estimate of miss rate based on amplitude histogram.
            isolation_distance (float): Distance to nearest cluster.
            l_ratio (float): Amount of empty space between a cluster and other spikes in dataset.
            d_prime (float): Classification accuracy based on LDA.
            nn_hit_rate (float): Fraction of neighbors for target cluster that are also in target cluster.
            nn_miss_rate (float): Fraction of neighbors outside target cluster that are in the target cluster.
            silhouette_core (float): Maximum change in spike depth throughout recording.
            cumulative_drift (float): Cumulative change in spike depth throughout recording.
            contamination_rate (float): Frequency of spikes in the refractory period.
        """

        definition = """
        # Cluster metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        firing_rate=null: float # (Hz) firing rate for a unit
        snr=null: float  # signal-to-noise ratio for a unit
        presence_ratio=null: float  # fraction of time in which spikes are present
        isi_violation=null: float   # rate of ISI violation as a fraction of overall rate
        number_violation=null: int  # total number of ISI violations
        amplitude_cutoff=null: float  # estimate of miss rate based on amplitude histogram
        isolation_distance=null: float  # distance to nearest cluster in Mahalanobis space
        l_ratio=null: float  #
        d_prime=null: float  # Classification accuracy based on LDA
        nn_hit_rate=null: float  # Fraction of neighbors for target cluster that are also in target cluster
        nn_miss_rate=null: float # Fraction of neighbors outside target cluster that are in target cluster
        silhouette_score=null: float  # Standard metric for cluster overlap
        max_drift=null: float  # Maximum change in spike depth throughout recording
        cumulative_drift=null: float  # Cumulative change in spike depth throughout recording
        contamination_rate=null: float #
        """

    class Waveform(dj.Part):
        """Waveform metrics for a particular unit.

        Attributes:
            QualityMetrics (foreign key): QualityMetrics primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            amplitude (float): Absolute difference between waveform peak and trough in microvolts.
            duration (float): Time between waveform peak and trough in milliseconds.
            halfwidth (float): Spike width at half max amplitude.
            pt_ratio (float): Absolute amplitude of peak divided by absolute amplitude of trough relative to 0.
            repolarization_slope (float): Slope of the regression line fit to first 30 microseconds from trough to peak.
            recovery_slope (float): Slope of the regression line fit to first 30 microseconds from peak to tail.
            spread (float): The range with amplitude over 12-percent of maximum amplitude along the probe.
            velocity_above (float): inverse velocity of waveform propagation from soma to the top of the probe.
            velocity_below (float): inverse velocity of waveform propagation from soma toward the bottom of the probe.
        """

        definition = """
        # Waveform metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        amplitude: float  # (uV) absolute difference between waveform peak and trough
        duration: float  # (ms) time between waveform peak and trough
        halfwidth=null: float  # (ms) spike width at half max amplitude
        pt_ratio=null: float  # absolute amplitude of peak divided by absolute amplitude of trough relative to 0
        repolarization_slope=null: float  # the repolarization slope was defined by fitting a regression line to the first 30us from trough to peak
        recovery_slope=null: float  # the recovery slope was defined by fitting a regression line to the first 30us from peak to tail
        spread=null: float  # (um) the range with amplitude above 12-percent of the maximum amplitude along the probe
        velocity_above=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the top of the probe
        velocity_below=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the bottom of the probe
        """

    def make(self, key):
        """Populates tables with quality metrics data."""
        output_dir = (ClusteringTask & key).fetch1("clustering_output_dir")
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        metric_fp = kilosort_dir / "metrics.csv"

        if not metric_fp.exists():
            raise FileNotFoundError(f"QC metrics file not found: {metric_fp}")

        metrics_df = pd.read_csv(metric_fp)
        metrics_df.set_index("cluster_id", inplace=True)
        metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics_list = [
            dict(metrics_df.loc[unit_key["unit"]], **unit_key)
            for unit_key in (CuratedClustering.Unit & key).fetch("KEY")
        ]

        self.insert1(key)
        self.Cluster.insert(metrics_list, ignore_extra_fields=True)
        self.Waveform.insert(metrics_list, ignore_extra_fields=True)


# ---------------- HELPER FUNCTIONS ----------------


# def get_spikeglx_meta_filepath(ephys_recording_key: dict) -> str:
#     """Get spikeGLX data filepath."""
#     # attempt to retrieve from EphysSession.EphysFile
#     spikeglx_meta_filepath = pathlib.Path(
#         (
#             EphysSession.EphysFile
#             & ephys_recording_key
#             & 'file_path LIKE "%.ap.meta"'
#         ).fetch1("file_path")
#     )

#     try:
#         spikeglx_meta_filepath = find_full_path(
#             get_ephys_root_data_dir(), spikeglx_meta_filepath
#         )
#     except FileNotFoundError:
#         # if not found, search in session_dir again
#         if not spikeglx_meta_filepath.exists():
#             session_dir = find_full_path(
#                 get_ephys_root_data_dir(), get_session_directory(ephys_recording_key)
#             )
#             inserted_probe_serial_number = (
#                 ProbeInsertion * probe.Probe & ephys_recording_key
#             ).fetch1("probe")

#             spikeglx_meta_filepaths = [fp for fp in session_dir.rglob("*.ap.meta")]
#             for meta_filepath in spikeglx_meta_filepaths:
#                 spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
#                 if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
#                     spikeglx_meta_filepath = meta_filepath
#                     break
#             else:
#                 raise FileNotFoundError(
#                     "No SpikeGLX data found for probe insertion: {}".format(
#                         ephys_recording_key
#                     )
#                 )

#     return spikeglx_meta_filepath


def get_openephys_probe_data(ephys_recording_key: dict) -> list:
    """Get OpenEphys probe data from file."""
    inserted_probe_serial_number = (
        ProbeInsertion * probe.Probe & ephys_recording_key
    ).fetch1("probe")
    session_dir = find_full_path(
        get_ephys_root_data_dir(), get_session_directory(ephys_recording_key)
    )
    loaded_oe = openephys.OpenEphys(session_dir)
    probe_data = loaded_oe.probes[inserted_probe_serial_number]

    # explicitly garbage collect "loaded_oe"
    # as these may have large memory footprint and may not be cleared fast enough
    del loaded_oe
    gc.collect()

    return probe_data


def get_neuropixels_channel2electrode_map(
    ephys_recording_key: dict, acq_software: str
) -> dict:
    """Get the channel map for neuropixels probe."""
    if acq_software == "SpikeGLX":
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_meta = spikeglx.SpikeGLXMeta(spikeglx_meta_filepath)
        electrode_config_key = (
            EphysSession * probe.ElectrodeConfig & ephys_recording_key
        ).fetch1("KEY")

        electrode_query = (
            probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode
            & electrode_config_key
        )

        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(
                *electrode_query.fetch("KEY", "shank", "shank_col", "shank_row")
            )
        }

        channel2electrode_map = {
            recorded_site: probe_electrodes[(shank, shank_col, shank_row)]
            for recorded_site, (shank, shank_col, shank_row, _) in enumerate(
                spikeglx_meta.shankmap["data"]
            )
        }
    elif acq_software == "Open Ephys":
        probe_dataset = get_openephys_probe_data(ephys_recording_key)

        electrode_query = (
            probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode * EphysSession
            & ephys_recording_key
        )

        probe_electrodes = {
            key["electrode"]: key for key in electrode_query.fetch("KEY")
        }

        channel2electrode_map = {
            channel_idx: probe_electrodes[channel_idx]
            for channel_idx in probe_dataset.ap_meta["channels_indices"]
        }

    return channel2electrode_map
