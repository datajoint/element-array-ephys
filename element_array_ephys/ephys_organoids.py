import importlib
import inspect
import pathlib
from datetime import datetime
from decimal import Decimal

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory
from scipy import signal

import intanrhdreader

from . import ephys_report, probe
from .readers import kilosort, openephys, spikeglx
from .spike_sorting import si_spike_sorting as ephys_sorter

schema = dj.schema()

logger = dj.logger

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
        culture.Experiment: A parent table to EphysSession.

    Functions:
        get_ephys_root_data_dir(): Returns absolute path for root data director(y/ies) with all electrophysiological recording sessions, as a list of string(s).
        get_organoid_directory(session_key: dict): Returns path to electrophysiology data for the a particular session as a list of strings.
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


def get_organoid_directory(session_key: dict) -> str:
    """Retrieve the session directory with Neuropixels for the given session.

    Args:
        session_key (dict): A dictionary mapping subject to an entry in the subject table, and session_datetime corresponding to a session in the database.

    Returns:
        A string for the path to the session directory.
    """
    return _linking_module.get_organoid_directory(session_key)


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
class Port(dj.Lookup):
    definition = """  # Port ID of the Intan acquisition system
    port_id     : char(2)
    """
    contents = zip(["A", "B", "C", "D"])


@schema
class EphysRawFile(dj.Manual):
    definition = """ # Catalog of all raw ephys files
    file_path         : varchar(512) # path to the file relative to the root directory
    ---
    -> AcquisitionSoftware
    file_time         : datetime #  date and time of the file acquisition
    parent_folder     : varchar(128) #  parent folder containing the file
    filename_prefix   : varchar(64)  #  filename prefix, if any, excluding the datetime information
    """


@schema
class EphysSession(dj.Manual):
    definition = """ # User defined ephys session for downstream analysis.
    -> culture.Experiment
    insertion_number            : tinyint unsigned
    start_time                  : datetime
    end_time                    : datetime
    ---
    session_type                : enum("lfp", "spike_sorting", "both")
    """


@schema
class EphysSessionProbe(dj.Manual):
    """User defined probe for each ephys session.
    Attributes:
        EphysSession (foreign key): EphysSession primary key.
        probe.Probe (foreign key): probe.Probe primary key.
        probe.ElectrodeConfig (foreign key): probe.ElectrodeConfig primary key.
    """

    definition = """
    -> EphysSession
    ---
    -> probe.Probe 
    -> Port  # port ID where the probe was connected to.
    used_electrodes=null     : longblob  # list of electrode IDs used in this session (if null, all electrodes are used)
    """


@schema
class EphysSessionInfo(dj.Imported):
    definition = """ # Store header information from the first session file.
    -> EphysSession
    ---
    session_info: longblob  # Session header info from intan .rhd file. Get this from the first session file.
    """

    def make(self, key):
        query = (
            EphysRawFile
            & f"file_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
        )
        if not query:
            raise FileNotFoundError(
                f"No EphysRawFile found BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
            )

        first_file = query.fetch("file_path", order_by="file_time", limit=1)[0]

        first_file = find_full_path(get_ephys_root_data_dir(), first_file)

        # Read file header
        with open(first_file, "rb") as f:
            header = intanrhdreader.read_header(f)
            del header["spike_triggers"], header["aux_input_channels"]

        logger.info(f"Populating ephys.EphysSessionInfo for <{key}>")

        self.insert(
            [
                {
                    **key,
                    "session_info": header,
                }
            ]
        )


@schema
class LFP(dj.Imported):
    definition = """ # Store pre-processed LFP traces per electrode. Only the LFPs collected from a pre-defined recording session.
    -> EphysSession
    ---
    lfp_sampling_rate    : float # Down-sampled sampling rate (Hz).
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode
        ---
        lfp              : blob@datajoint-blob
        """

    @property
    def key_source(self):
        return (EphysSession & EphysSessionProbe) - "session_type='spike_sorting'"

    def make(self, key):
        TARGET_SAMPLING_RATE = 2500  # Hz
        POWERLINE_NOISE_FREQ = 60  # Hz
        LFP_DURATION = 30  # minutes

        start_time = datetime.strptime(key["start_time"], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(key["end_time"], "%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds() / 60  # minutes

        assert (
            duration <= LFP_DURATION
        ), f"LFP sessions cannot exceeds {LFP_DURATION} minutes in duration."

        query = (
            EphysRawFile
            & f"file_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
        )
        if not query:
            logger.info(
                f"No raw data file found. Skip populating ephys.LFP for <{key}>"
            )
        else:
            logger.info(f"Populating ephys.LFP for <{key}>")

            # Get probe info
            probe_info = (EphysSessionProbe & key).fetch1()
            probe_type = (probe.Probe & {"probe": probe_info["probe"]}).fetch1(
                "probe_type"
            )

            electrode_query = probe.ElectrodeConfig.Electrode & (
                probe.ElectrodeConfig & {"probe_type": probe_type}
            )

            # Filter for used electrodes. If probe_info["used_electrodes"] is None, it means all electrodes were used.
            if probe_info["used_electrodes"]:
                electrode_query &= (
                    f'electrode IN {tuple(probe_info["used_electrodes"])}'
                )

            header = {}
            lfp_concat = np.array([], dtype=np.float64)

            for file_relpath in query.fetch("file_path", order_by="file_time"):
                file = find_full_path(get_ephys_root_data_dir(), file_relpath)

                try:
                    data = intanrhdreader.load_file(file)
                except OSError:
                    raise OSError(f"OS error occured when loading file {file.name}")

                if not header:
                    header = data.pop("header")
                    lfp_sampling_rate = header["sample_rate"]
                    powerline_noise_freq = (
                        header["notch_filter_frequency"] or POWERLINE_NOISE_FREQ
                    )  # in Hz
                    downsample_factor = int(lfp_sampling_rate / TARGET_SAMPLING_RATE)

                    # Get LFP indices (row index of the LFP matrix to be used)
                    lfp_indices = np.array(electrode_query.fetch("channel"), dtype=int)
                    port_indices = np.array(
                        [
                            ind
                            for ind, ch in enumerate(data["amplifier_channels"])
                            if ch["port_prefix"] == probe_info["port_id"]
                        ]
                    )
                    lfp_indices = np.sort(port_indices[lfp_indices])

                    self.insert1(
                        {
                            **key,
                            "lfp_sampling_rate": TARGET_SAMPLING_RATE,
                        }
                    )

                    channels = np.array(
                        [
                            ch["native_channel_name"]
                            for ch in data["amplifier_channels"]
                            if ch["port_prefix"]
                        ]
                    )[lfp_indices]

                    electrode_df = electrode_query.fetch(format="frame").reset_index()

                    channel_to_electrode_map = dict(
                        zip(electrode_df["channel"], electrode_df["electrode"])
                    )

                    channel_to_electrode_map = {
                        f'{probe_info["port_id"]}-{int(channel):03d}': electrode
                        for channel, electrode in channel_to_electrode_map.items()
                    }

                lfps = data.pop("amplifier_data")[lfp_indices]
                lfp_concat = (
                    lfps if lfp_concat.size == 0 else np.hstack((lfp_concat, lfps))
                )
                del data

            # Check for missing files or short trace durations in min
            trace_duration = lfp_concat.shape[1] / TARGET_SAMPLING_RATE / 60  # in min
            if trace_duration != (EphysSession & key).proj(
                duration="TIMESTAMPDIFF(MINUTE, start_time, end_time)"
            ).fetch1("duration"):
                raise ValueError(
                    f"Trace legnth ({trace_duration} min) is less than session duration"
                )

            # Single insert in loop to mitigate potential memory issue.
            for ch, lfp in zip(channels, lfp_concat):
                # Powerline noise removal
                b_notch, a_notch = signal.iirnotch(
                    w0=powerline_noise_freq, Q=30, fs=TARGET_SAMPLING_RATE
                )
                lfp = signal.filtfilt(b_notch, a_notch, lfp)

                # Lowpass filter
                b_butter, a_butter = signal.butter(
                    N=4, Wn=1000, btype="lowpass", fs=TARGET_SAMPLING_RATE
                )
                lfp = signal.filtfilt(b_butter, a_butter, lfp)

                # Downsample the signal
                lfp = lfp[::downsample_factor]

                self.Trace.insert1(
                    {
                        **key,
                        "electrode_config_hash": electrode_df["electrode_config_hash"][
                            0
                        ],
                        "probe_type": electrode_df["probe_type"][0],
                        "electrode": channel_to_electrode_map[ch],
                        "lfp": lfp,
                    }
                )


# ------------ Clustering --------------


@schema
class ClusteringMethod(dj.Lookup):
    """Kilosort clustering method.

    Attributes:
        clustering_method (foreign key, varchar(20) ): Kilosort clustering method.
        clustering_methods_desc (varchar(1000) ): Additional description of the clustering method.
    """

    definition = """
    # Method for clustering
    clustering_method: varchar(20)
    ---
    clustering_method_desc: varchar(1000)
    """
    contents = [(sorter, "") for sorter in ephys_sorter.SI_SORTERS]


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
        ("n.a.", "not available"),
    ]


@schema
class ClusteringTask(dj.Manual):
    """A clustering task to spike sort electrophysiology datasets.

    Attributes:
        EphysSession (foreign key): EphysSession primary key.
        ClusteringParamSet (foreign key): ClusteringParamSet primary key.
        clustering_outdir_dir (varchar (255) ): Relative path to output clustering results.
    """

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysSession
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
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
                processed_dir / subject_dir / {clustering_method}_{paramset_idx}
                e.g.: sub4/sess1/kilosort2_0
        """
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        exp_dir = find_full_path(get_ephys_root_data_dir(), get_organoid_directory(key))

        session_time = "_".join(
            [
                key["start_time"].strftime("%Y%m%d%H%M"),
                key["end_time"].strftime("%Y%m%d%H%M"),
            ]
        )

        session_dir = exp_dir / session_time
        root_dir = find_root_directory(get_ephys_root_data_dir(), exp_dir)

        method = (
            (ClusteringParamSet * ClusteringMethod & key)
            .fetch1("clustering_method")
            .replace(".", "-")
        )

        output_dir = (
            processed_dir
            / session_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{output_dir} created!")

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

        cls.insert1(
            {
                **key,
                "clustering_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
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
        """This will be implemented via si_spike_sorting tables."""
        pass


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
            else (
                "spike_times_sec"
                if "spike_times_sec" in kilosort_dataset.data
                else "spike_times"
            )
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
                        "spike_depths": (
                            spike_depths[
                                kilosort_dataset.data["spike_clusters"] == unit
                            ]
                            if spike_depths is not None
                            else None
                        ),
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
                subject_dir = find_full_path(
                    get_ephys_root_data_dir(), get_organoid_directory(key)
                )
                openephys_dataset = openephys.OpenEphys(subject_dir)
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
