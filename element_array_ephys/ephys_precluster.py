import importlib
import inspect
import re

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory

from . import ephys_report, probe
from .readers import kilosort, openephys, spikeglx

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
        Session: A parent table to ProbeInsertion
        Probe: A parent table to EphysRecording. Probe information is required before electrophysiology data is imported.

    Functions:
        get_ephys_root_data_dir(): Returns absolute path for root data director(y/ies) with all electrophysiological recording sessions, as a list of string(s).
        get_session_direction(session_key: dict): Returns path to electrophysiology data for the a particular session as a list of strings.
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
    return _linking_module.get_ephys_root_data_dir()


def get_session_directory(session_key: dict) -> str:
    """Retrieve the session directory with Neuropixels for the given session.

    Args:
        session_key (dict): A dictionary mapping subject to an entry in the subject table, and session_datetime corresponding to a session in the database.

    Returns:
        A string for the path to the session directory.
    """
    return _linking_module.get_session_directory(session_key)


# ----------------------------- Table declarations ----------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    """Name of software used for recording electrophysiological data.

    Attributes:
        acq_software ( varchar(24) ): Acquisition software, e.g,. SpikeGLX, OpenEphys
    """

    definition = """  # Name of software used for recording of neuropixels probes - SpikeGLX or Open Ephys
    acq_software: varchar(24)
    """
    contents = zip(["SpikeGLX", "Open Ephys"])


@schema
class ProbeInsertion(dj.Manual):
    """Information about probe insertion across subjects and sessions.

    Attributes:
        Session (foreign key): Session primary key.
        insertion_number (foreign key, str): Unique insertion number for each probe insertion for a given session.
        probe.Probe (str): probe.Probe primary key.
    """

    definition = """
    # Probe insertion implanted into an animal for a given session.
    -> Session
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """


@schema
class InsertionLocation(dj.Manual):
    """Stereotaxic location information for each probe insertion.

    Attributes:
        ProbeInsertion (foreign key): ProbeInsertion primary key.
        SkullReference (dict): SkullReference primary key.
        ap_location (decimal (6, 2) ): Anterior-posterior location in micrometers. Reference is 0 with anterior values positive.
        ml_location (decimal (6, 2) ): Medial-lateral location in micrometers. Reference is zero with right side values positive.
        depth (decimal (6, 2) ): Manipulator depth relative to the surface of the brain at zero. Ventral is negative.
        Theta (decimal (5, 2) ): elevation - rotation about the ml-axis in degrees relative to positive z-axis.
        phi (decimal (5, 2) ): azimuth - rotation about the dv-axis in degrees relative to the positive x-axis

    """

    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    -> SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysRecording(dj.Imported):
    """Automated table with electrophysiology recording information for each probe inserted during an experimental session.

    Attributes:
        ProbeInsertion (foreign key): ProbeInsertion primary key.
        probe.ElectrodeConfig (dict): probe.ElectrodeConfig primary key.
        AcquisitionSoftware (dict): AcquisitionSoftware primary key.
        sampling_rate (float): sampling rate of the recording in Hertz (Hz).
        recording_datetime (datetime): datetime of the recording from this probe.
        recording_duration (float): duration of the entire recording from this probe in seconds.
    """

    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion
    ---
    -> probe.ElectrodeConfig
    -> AcquisitionSoftware
    sampling_rate: float # (Hz)
    recording_datetime: datetime # datetime of the recording from this probe
    recording_duration: float # (seconds) duration of the recording from this probe
    """

    class EphysFile(dj.Part):
        """Paths of electrophysiology recording files for each insertion.

        Attributes:
            EphysRecording (foreign key): EphysRecording primary key.
            file_path (varchar(255) ): relative file path for electrophysiology recording.
        """

        definition = """
        # Paths of files of a given EphysRecording round.
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):
        """Populates table with electrophysiology recording information."""
        session_dir = find_full_path(
            get_ephys_root_data_dir(), get_session_directory(key)
        )

        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1(
            "probe"
        )

        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in (
            ("*.ap.meta", "SpikeGLX"),
            ("*.oebin", "Open Ephys"),
        ):
            ephys_meta_filepaths = [fp for fp in session_dir.rglob(ephys_pattern)]
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f"Ephys recording data not found!"
                f" Neither SpikeGLX nor Open Ephys recording files found"
                f" in {session_dir}"
            )

        if acq_software == "SpikeGLX":
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    "No SpikeGLX data found for probe insertion: {}".format(key)
                )

            if re.search("(1.0|2.0)", spikeglx_meta.probe_model):
                probe_type = spikeglx_meta.probe_model
                electrode_query = probe.ProbeType.Electrode & {"probe_type": probe_type}

                probe_electrodes = {
                    (shank, shank_col, shank_row): key
                    for key, shank, shank_col, shank_row in zip(
                        *electrode_query.fetch("KEY", "shank", "shank_col", "shank_row")
                    )
                }

                electrode_group_members = [
                    probe_electrodes[(shank, shank_col, shank_row)]
                    for shank, shank_col, shank_row, _ in spikeglx_meta.shankmap["data"]
                ]
            else:
                raise NotImplementedError(
                    "Processing for neuropixels probe model"
                    " {} not yet implemented".format(spikeglx_meta.probe_model)
                )

            self.insert1(
                {
                    **key,
                    **generate_electrode_config(probe_type, electrode_group_members),
                    "acq_software": acq_software,
                    "sampling_rate": spikeglx_meta.meta["imSampRate"],
                    "recording_datetime": spikeglx_meta.recording_time,
                    "recording_duration": (
                        spikeglx_meta.recording_duration
                        or spikeglx.retrieve_recording_duration(meta_filepath)
                    ),
                }
            )

            root_dir = find_root_directory(get_ephys_root_data_dir(), meta_filepath)
            self.EphysFile.insert1(
                {**key, "file_path": meta_filepath.relative_to(root_dir).as_posix()}
            )
        elif acq_software == "Open Ephys":
            dataset = openephys.OpenEphys(session_dir)
            for serial_number, probe_data in dataset.probes.items():
                if str(serial_number) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    "No Open Ephys data found for probe insertion: {}".format(key)
                )

            if re.search("(1.0|2.0)", probe_data.probe_model):
                probe_type = probe_data.probe_model
                electrode_query = probe.ProbeType.Electrode & {"probe_type": probe_type}

                probe_electrodes = {
                    key["electrode"]: key for key in electrode_query.fetch("KEY")
                }

                electrode_group_members = [
                    probe_electrodes[channel_idx]
                    for channel_idx in probe_data.ap_meta["channels_ids"]
                ]
            else:
                raise NotImplementedError(
                    "Processing for neuropixels"
                    " probe model {} not yet implemented".format(probe_data.probe_model)
                )

            self.insert1(
                {
                    **key,
                    **generate_electrode_config(probe_type, electrode_group_members),
                    "acq_software": acq_software,
                    "sampling_rate": probe_data.ap_meta["sample_rate"],
                    "recording_datetime": probe_data.recording_info[
                        "recording_datetimes"
                    ][0],
                    "recording_duration": np.sum(
                        probe_data.recording_info["recording_durations"]
                    ),
                }
            )

            root_dir = find_root_directory(
                get_ephys_root_data_dir(),
                probe_data.recording_info["recording_files"][0],
            )
            self.EphysFile.insert(
                [
                    {**key, "file_path": fp.relative_to(root_dir).as_posix()}
                    for fp in probe_data.recording_info["recording_files"]
                ]
            )
        else:
            raise NotImplementedError(
                f"Processing ephys files from"
                f" acquisition software of type {acq_software} is"
                f" not yet implemented"
            )


@schema
class PreClusterMethod(dj.Lookup):
    """Pre-clustering method

    Attributes:
        precluster_method (foreign key, varchar(16) ): Pre-clustering method for the dataset.
        precluster_method_desc(varchar(1000) ): Pre-clustering method description.
    """

    definition = """
    # Method for pre-clustering
    precluster_method: varchar(16)
    ---
    precluster_method_desc: varchar(1000)
    """

    contents = [("catgt", "Time shift, Common average referencing, Zeroing")]


@schema
class PreClusterParamSet(dj.Lookup):
    """Parameters for the pre-clustering method.

    Attributes:
        paramset_idx (foreign key): Unique parameter set ID.
        PreClusterMethod (dict): PreClusterMethod query for this dataset.
        paramset_desc (varchar(128) ): Description for the pre-clustering parameter set.
        param_set_hash (uuid): Unique hash for parameter set.
        params (longblob): All parameters for the pre-clustering method.
    """

    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> PreClusterMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(
        cls, precluster_method: str, paramset_idx: int, paramset_desc: str, params: dict
    ):
        param_dict = {
            "precluster_method": precluster_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
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
                    "The specified param-set"
                    " already exists - paramset_idx: {}".format(existing_paramset_idx)
                )
        else:
            cls.insert1(param_dict)


@schema
class PreClusterParamSteps(dj.Manual):
    """Ordered list of parameter sets that will be run.

    Attributes:
        precluster_param_steps_id (foreign key): Unique ID for the pre-clustering parameter sets to be run.
        precluster_param_steps_name (varchar(32) ): User-friendly name for the parameter steps.
        precluster_param_steps_desc (varchar(128) ): Description of the parameter steps.
    """

    definition = """
    # Ordered list of paramset_idx that are to be run
    # When pre-clustering is not performed, do not create an entry in `Step` Part table
    precluster_param_steps_id: smallint
    ---
    precluster_param_steps_name: varchar(32)
    precluster_param_steps_desc: varchar(128)
    """

    class Step(dj.Part):
        """Define the order of operations for parameter sets.

        Attributes:
            PreClusterParamSteps (foreign key): PreClusterParamSteps primary key.
            step_number (foreign key, smallint): Order of operations.
            PreClusterParamSet (dict): PreClusterParamSet to be used in pre-clustering.
        """

        definition = """
        -> master
        step_number: smallint                  # Order of operations
        ---
        -> PreClusterParamSet
        """


@schema
class PreClusterTask(dj.Manual):
    """Defines a pre-clustering task ready to be run.

    Attributes:
        EphysRecording (foreign key): EphysRecording primary key.
        PreclusterParamSteps (foreign key): PreClusterParam Steps primary key.
        precluster_output_dir (varchar(255) ): relative path to directory for storing results of pre-clustering.
        task_mode (enum ): `none` (no pre-clustering), `load` results from file, or `trigger` automated pre-clustering.
    """

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysRecording
    -> PreClusterParamSteps
    ---
    precluster_output_dir='': varchar(255)  #  pre-clustering output directory relative to the root data directory
    task_mode='none': enum('none','load', 'trigger') # 'none': no pre-clustering analysis
                                                     # 'load': load analysis results
                                                     # 'trigger': trigger computation
    """


@schema
class PreCluster(dj.Imported):
    """
    A processing table to handle each PreClusterTask:

    Attributes:
        PreClusterTask (foreign key): PreClusterTask primary key.
        precluster_time (datetime): Time of generation of this set of pre-clustering results.
        package_version (varchar(16) ): Package version used for performing pre-clustering.
    """

    definition = """
    -> PreClusterTask
    ---
    precluster_time: datetime  # time of generation of this set of pre-clustering results
    package_version='': varchar(16)
    """

    def make(self, key):
        """Populate pre-clustering tables."""
        task_mode, output_dir = (PreClusterTask & key).fetch1(
            "task_mode", "precluster_output_dir"
        )
        precluster_output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == "none":
            if len((PreClusterParamSteps.Step & key).fetch()) > 0:
                raise ValueError(
                    "There are entries in the PreClusterParamSteps.Step "
                    "table and task_mode=none"
                )
            creation_time = (EphysRecording & key).fetch1("recording_datetime")
        elif task_mode == "load":
            acq_software = (EphysRecording & key).fetch1("acq_software")
            inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1(
                "probe"
            )

            if acq_software == "SpikeGLX":
                for meta_filepath in precluster_output_dir.rglob("*.ap.meta"):
                    spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)

                    if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                        creation_time = spikeglx_meta.recording_time
                        break
                else:
                    raise FileNotFoundError(
                        "No SpikeGLX data found for probe insertion: {}".format(key)
                    )
            else:
                raise NotImplementedError(
                    f"Pre-clustering analysis of {acq_software}" "is not yet supported."
                )
        elif task_mode == "trigger":
            raise NotImplementedError(
                "Automatic triggering of"
                " pre-clustering analysis is not yet supported."
            )
        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        self.insert1({**key, "precluster_time": creation_time, "package_version": ""})


@schema
class LFP(dj.Imported):
    """Extracts local field potentials (LFP) from an electrophysiology recording.

    Attributes:
        EphysRecording (foreign key): EphysRecording primary key.
        lfp_sampling_rate (float): Sampling rate for LFPs in Hz.
        lfp_time_stamps (longblob): Time stamps with respect to the start of the recording.
        lfp_mean (longblob): Overall mean LFP across electrodes.
    """

    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> PreCluster
    ---
    lfp_sampling_rate: float   # (Hz)
    lfp_time_stamps: longblob  # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob         # (uV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        """Saves local field potential data for each electrode.

        Attributes:
            LFP (foreign key): LFP primary key.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            lfp (longblob): LFP recording at this electrode in microvolts.
        """

        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode
        ---
        lfp: longblob               # (uV) recorded lfp at this electrode
        """

    # Only store LFP for every 9th channel, due to high channel density,
    # close-by channels exhibit highly similar LFP
    _skip_channel_counts = 9

    def make(self, key):
        """Populates the LFP tables."""
        acq_software, probe_sn = (EphysRecording * ProbeInsertion & key).fetch1(
            "acq_software", "probe"
        )

        electrode_keys, lfp = [], []

        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)

            lfp_channel_ind = spikeglx_recording.lfmeta.recording_channels[
                -1 :: -self._skip_channel_counts
            ]

            # Extract LFP data at specified channels and convert to uV
            lfp = spikeglx_recording.lf_timeseries[
                :, lfp_channel_ind
            ]  # (sample x channel)
            lfp = (
                lfp * spikeglx_recording.get_channel_bit_volts("lf")[lfp_channel_ind]
            ).T  # (channel x sample)

            self.insert1(
                dict(
                    key,
                    lfp_sampling_rate=spikeglx_recording.lfmeta.meta["imSampRate"],
                    lfp_time_stamps=(
                        np.arange(lfp.shape[1])
                        / spikeglx_recording.lfmeta.meta["imSampRate"]
                    ),
                    lfp_mean=lfp.mean(axis=0),
                )
            )

            electrode_query = (
                probe.ProbeType.Electrode
                * probe.ElectrodeConfig.Electrode
                * EphysRecording
                & key
            )
            probe_electrodes = {
                (shank, shank_col, shank_row): key
                for key, shank, shank_col, shank_row in zip(
                    *electrode_query.fetch("KEY", "shank", "shank_col", "shank_row")
                )
            }

            for recorded_site in lfp_channel_ind:
                shank, shank_col, shank_row, _ = spikeglx_recording.apmeta.shankmap[
                    "data"
                ][recorded_site]
                electrode_keys.append(probe_electrodes[(shank, shank_col, shank_row)])
        elif acq_software == "Open Ephys":
            session_dir = find_full_path(
                get_ephys_root_data_dir(), get_session_directory(key)
            )

            loaded_oe = openephys.OpenEphys(session_dir)
            oe_probe = loaded_oe.probes[probe_sn]

            lfp_channel_ind = np.arange(len(oe_probe.lfp_meta["channels_ids"]))[
                -1 :: -self._skip_channel_counts
            ]

            lfp = oe_probe.lfp_timeseries[:, lfp_channel_ind]  # (sample x channel)
            lfp = (
                lfp * np.array(oe_probe.lfp_meta["channels_gains"])[lfp_channel_ind]
            ).T  # (channel x sample)
            lfp_timestamps = oe_probe.lfp_timestamps

            self.insert1(
                dict(
                    key,
                    lfp_sampling_rate=oe_probe.lfp_meta["sample_rate"],
                    lfp_time_stamps=lfp_timestamps,
                    lfp_mean=lfp.mean(axis=0),
                )
            )

            electrode_query = (
                probe.ProbeType.Electrode
                * probe.ElectrodeConfig.Electrode
                * EphysRecording
                & key
            )
            probe_electrodes = {
                key["electrode"]: key for key in electrode_query.fetch("KEY")
            }

            for channel_idx in np.array(oe_probe.lfp_meta["channels_ids"])[
                lfp_channel_ind
            ]:
                electrode_keys.append(probe_electrodes[channel_idx])
        else:
            raise NotImplementedError(
                f"LFP extraction from acquisition software"
                f" of type {acq_software} is not yet implemented"
            )

        # single insert in loop to mitigate potential memory issue
        for electrode_key, lfp_trace in zip(electrode_keys, lfp):
            self.Electrode.insert1({**key, **electrode_key, "lfp": lfp_trace})


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
        ("kilosort", "kilosort clustering method"),
        ("kilosort2", "kilosort2 clustering method"),
    ]


@schema
class ClusteringParamSet(dj.Lookup):
    """Parameters to be used in clustering procedure for spike sorting.

    Attributes:
        paramset_idx (foreign key): Unique ID for the clustering parameter set.
        ClusteringMethod (dict): ClusteringMethod primary key.
        paramset_desc (varchar(128) ): Description of the clustering parameter set.
        param_set_hash (uuid): UUID hash for the parameter set.
        params (longblob): Paramset, dictionary of all applicable parameters.
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
        cls, processing_method: str, paramset_idx: int, paramset_desc: str, params: dict
    ):
        """Inserts new parameters into the ClusteringParamSet table.

        Args:
            processing_method (str): name of the clustering method.
            paramset_desc (str): description of the parameter set
            params (dict): clustering parameters
            paramset_idx (int, optional): Unique parameter set ID. Defaults to None.
        """
        param_dict = {
            "clustering_method": processing_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
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
                    "The specified param-set"
                    " already exists - paramset_idx: {}".format(existing_paramset_idx)
                )
        else:
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
    cluster_quality_label:  varchar(100)
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
        EphysRecording (foreign key): EphysRecording primary key.
        ClusteringParamSet (foreign key): ClusteringParamSet primary key.
        clustering_outdir_dir (varchar (255) ): Relative path to output clustering results.
        task_mode (enum): `Trigger` computes clustering or and `load` imports existing data.
    """

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> PreCluster
    -> ClusteringParamSet
    ---
    clustering_output_dir: varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


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
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == "load":
            _ = kilosort.Kilosort(
                kilosort_dir
            )  # check if the directory is a valid Kilosort output
            creation_time, _, _ = kilosort.extract_clustering_info(kilosort_dir)
        elif task_mode == "trigger":
            raise NotImplementedError(
                "Automatic triggering of" " clustering analysis is not yet supported"
            )
        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        self.insert1({**key, "clustering_time": creation_time, "package_version": ""})


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
            spike_times (longblob): Spike times of this unit, relative to start time of EphysRecording.
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
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe
        """

    def make(self, key):
        """Automated population of Unit information."""
        output_dir = (Curation & key).fetch1("curation_output_dir")
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)
        acq_software = (EphysRecording & key).fetch1("acq_software")

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
                    / kilosort_dataset.data["params"]["sample_rate"]
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
            EphysRecording * ProbeInsertion & key
        ).fetch1("acq_software", "probe")

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1("KEY")
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
            self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
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
        rename_dict = {
            "isi_viol": "isi_violation",
            "num_viol": "number_violation",
            "contam_rate": "contamination_rate",
        }

        if not metric_fp.exists():
            raise FileNotFoundError(f"QC metrics file not found: {metric_fp}")

        metrics_df = pd.read_csv(metric_fp)
        metrics_df.set_index("cluster_id", inplace=True)
        metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics_df.columns = metrics_df.columns.str.lower()
        metrics_df.rename(columns=rename_dict, inplace=True)
        metrics_list = [
            dict(metrics_df.loc[unit_key["unit"]], **unit_key)
            for unit_key in (CuratedClustering.Unit & key).fetch("KEY")
        ]

        self.insert1(key)
        self.Cluster.insert(metrics_list, ignore_extra_fields=True)
        self.Waveform.insert(metrics_list, ignore_extra_fields=True)


# ---------------- HELPER FUNCTIONS ----------------


def get_spikeglx_meta_filepath(ephys_recording_key: dict) -> str:
    """Get spikeGLX data filepath."""
    # attempt to retrieve from EphysRecording.EphysFile
    spikeglx_meta_filepath = (
        EphysRecording.EphysFile & ephys_recording_key & 'file_path LIKE "%.ap.meta"'
    ).fetch1("file_path")

    try:
        spikeglx_meta_filepath = find_full_path(
            get_ephys_root_data_dir(), spikeglx_meta_filepath
        )
    except FileNotFoundError:
        # if not found, search in session_dir again
        if not spikeglx_meta_filepath.exists():
            session_dir = find_full_path(
                get_ephys_root_data_dir(), get_session_directory(ephys_recording_key)
            )
            inserted_probe_serial_number = (
                ProbeInsertion * probe.Probe & ephys_recording_key
            ).fetch1("probe")

            spikeglx_meta_filepaths = [fp for fp in session_dir.rglob("*.ap.meta")]
            for meta_filepath in spikeglx_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    spikeglx_meta_filepath = meta_filepath
                    break
            else:
                raise FileNotFoundError(
                    "No SpikeGLX data found for probe insertion: {}".format(
                        ephys_recording_key
                    )
                )

    return spikeglx_meta_filepath


def get_neuropixels_channel2electrode_map(
    ephys_recording_key: dict, acq_software: str
) -> dict:
    """Get the channel map for neuropixels probe."""
    if acq_software == "SpikeGLX":
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_meta = spikeglx.SpikeGLXMeta(spikeglx_meta_filepath)
        electrode_config_key = (
            EphysRecording * probe.ElectrodeConfig & ephys_recording_key
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
        session_dir = find_full_path(
            get_ephys_root_data_dir(), get_session_directory(ephys_recording_key)
        )
        openephys_dataset = openephys.OpenEphys(session_dir)
        probe_serial_number = (ProbeInsertion & ephys_recording_key).fetch1("probe")
        probe_dataset = openephys_dataset.probes[probe_serial_number]

        electrode_query = (
            probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode * EphysRecording
            & ephys_recording_key
        )

        probe_electrodes = {
            key["electrode"]: key for key in electrode_query.fetch("KEY")
        }

        channel2electrode_map = {
            channel_idx: probe_electrodes[channel_idx]
            for channel_idx in probe_dataset.ap_meta["channels_ids"]
        }

    return channel2electrode_map


def generate_electrode_config(probe_type: str, electrode_keys: list) -> dict:
    """Generate and insert new ElectrodeConfig

    Args:
        probe_type (str): probe type (e.g. neuropixels 2.0 - SS)
        electrode_keys (list): list of keys of the probe.ProbeType.Electrode table

    Returns:
        dict: representing a key of the probe.ElectrodeConfig table
    """
    # compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode)
    electrode_config_hash = dict_to_uuid({k["electrode"]: k for k in electrode_keys})

    electrode_list = sorted([k["electrode"] for k in electrode_keys])
    electrode_gaps = (
        [-1]
        + np.where(np.diff(electrode_list) > 1)[0].tolist()
        + [len(electrode_list) - 1]
    )
    electrode_config_name = "; ".join(
        [
            f"{electrode_list[start + 1]}-{electrode_list[end]}"
            for start, end in zip(electrode_gaps[:-1], electrode_gaps[1:])
        ]
    )

    electrode_config_key = {"electrode_config_hash": electrode_config_hash}

    # ---- make new ElectrodeConfig if needed ----
    if not probe.ElectrodeConfig & electrode_config_key:
        probe.ElectrodeConfig.insert1(
            {
                **electrode_config_key,
                "probe_type": probe_type,
                "electrode_config_name": electrode_config_name,
            }
        )
        probe.ElectrodeConfig.Electrode.insert(
            {**electrode_config_key, **electrode} for electrode in electrode_keys
        )

    return electrode_config_key
