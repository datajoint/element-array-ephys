import os
import decimal
import json
import numpy as np
import pynwb
import datajoint as dj
from element_interface.utils import find_full_path
from hdmf.backends.hdf5 import H5DataIO
from hdmf.data_utils import GenericDataChunkIterator
from nwb_conversion_tools.tools.nwb_helpers import get_module
from nwb_conversion_tools.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator
)
from spikeinterface import extractors
from tqdm import tqdm
import warnings
from ... import probe, ephys_no_curation

assert probe.schema.is_activated(), 'probe not yet activated'

assert ephys_no_curation.schema.is_activated,  \
        "The ephys module must be activated before export."


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)


class LFPDataChunkIterator(GenericDataChunkIterator):
    """
    DataChunkIterator for LFP data that pulls data one channel at a time. Used when
    reading LFP data from the database (as opposed to directly from source files)
    """

    def __init__(self, lfp_electrodes_query, chunk_length: int = 10000):
        """
        Parameters
        ----------
        lfp_electrodes_query: element_array_ephys.ephys.LFP.Electrode
        chunk_length: int, optional
            Chunks are blocks of disk space where data are stored contiguously and compressed
        """
        self.lfp_electrodes_query = lfp_electrodes_query
        self.electrodes = self.lfp_electrodes_query.fetch("electrode")

        first_record = (
            self.lfp_electrodes_query & dict(electrode=self.electrodes[0])
        ).fetch1(as_dict=True)

        self.n_channels = len(self.electrodes)
        self.n_tt = len(first_record["lfp"])
        self._dtype = first_record["lfp"].dtype

        super().__init__(buffer_shape=(self.n_tt, 1), chunk_shape=(chunk_length, 1))

    def _get_data(self, selection):

        electrode = self.electrodes[selection[1]][0]
        return (self.lfp_electrodes_query & dict(electrode=electrode)).fetch1("lfp")[
            selection[0], np.newaxis
        ]

    def _get_dtype(self):
        return self._dtype

    def _get_maxshape(self):
        return self.n_tt, self.n_channels


def add_electrodes_to_nwb(session_key: dict, nwbfile: pynwb.NWBFile):
    """
    Add electrodes table to NWBFile. This is needed for any ElectricalSeries, including
    raw source data and LFP.

    ephys.InsertionLocation -> ElectrodeGroup.location

    probe.Probe::probe -> device.name
    probe.Probe::probe_comment -> device.description
    probe.Probe::probe_type -> device.manufacturer

    probe.ProbeType.Electrode::electrode -> electrodes["id_in_probe"]
    probe.ProbeType.Electrode::y_coord -> electrodes["rel_y"]
    probe.ProbeType.Electrode::x_coord -> electrodes["rel_x"]
    probe.ProbeType.Electrode::shank -> electrodes["shank"]
    probe.ProbeType.Electrode::shank_col -> electrodes["shank_col"]
    probe.ProbeType.Electrode::shank_row -> electrodes["shank_row"]

    Parameters
    ----------
    session_key: dict
    nwbfile: pynwb.NWBFile
    """
    electrodes_query = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode

    for additional_attribute in ["shank_col", "shank_row", "shank"]:
        nwbfile.add_electrode_column(
            name=electrodes_query.heading.attributes[additional_attribute].name,
            description=electrodes_query.heading.attributes[
                additional_attribute
            ].comment,
        )

    nwbfile.add_electrode_column(
        name="id_in_probe", description="electrode id within the probe",
    )

    for this_probe in (ephys.ProbeInsertion * probe.Probe & session_key).fetch(
        as_dict=True
    ):
        insertion_record = (ephys.InsertionLocation & this_probe).fetch(as_dict=True)
        if len(insertion_record) == 1:
            insert_location = json.dumps(
                {
                    k: v
                    for k, v in insertion_record[0].items()
                    if k not in ephys.InsertionLocation.primary_key
                },
                cls=DecimalEncoder,
            )
        elif len(insertion_record) == 0:
            insert_location = "unknown"
        else:
            raise DataJointError(f'Found multiple insertion locations for {this_probe}')

        device = nwbfile.create_device(
            name=this_probe["probe"],
            description=this_probe.get("probe_comment", None),
            manufacturer=this_probe.get("probe_type", None),
        )
        shank_ids = set((probe.ProbeType.Electrode & this_probe).fetch("shank"))
        for shank_id in shank_ids:
            electrode_group = nwbfile.create_electrode_group(
                name=f"probe{this_probe['probe']}_shank{shank_id}",
                description=f"probe{this_probe['probe']}_shank{shank_id}",
                location=insert_location,
                device=device,
            )

            electrodes_query = (
                probe.ProbeType.Electrode & this_probe & dict(shank=shank_id)
            ).fetch(as_dict=True)
            for electrode in electrodes_query:
                nwbfile.add_electrode(
                    group=electrode_group,
                    filtering="unknown",
                    imp=-1.0,
                    x=np.nan,
                    y=np.nan,
                    z=np.nan,
                    rel_x=electrode["x_coord"],
                    rel_y=electrode["y_coord"],
                    rel_z=np.nan,
                    shank_col=electrode["shank_col"],
                    shank_row=electrode["shank_row"],
                    location="unknown",
                    id_in_probe=electrode["electrode"],
                    shank=electrode["shank"],
                )


def create_units_table(
    session_key: dict,
    nwbfile: pynwb.NWBFile,
    paramset_record,
    name="units",
    desc="data on spiking units"):
    """

    ephys.CuratedClustering.Unit::unit -> units.id
    ephys.CuratedClustering.Unit::spike_times -> units["spike_times"]
    ephys.CuratedClustering.Unit::spike_depths -> units["spike_depths"]
    ephys.CuratedClustering.Unit::cluster_quality_label -> units["cluster_quality_label"]

    ephys.WaveformSet.PeakWaveform::peak_electrode_waveform -> units["waveform_mean"]

    Parameters
    ----------
    session_key: dict
    nwbfile: pynwb.NWBFile
    paramset_record: int
    name: str, optional
        default="units"
    desc: str, optional
        default="data on spiking units"
    """

    # electrode id mapping
    mapping = get_electrodes_mapping(nwbfile.electrodes)

    units_table = pynwb.misc.Units(name=name, description=desc)
    # add additional columns to the units table
    for additional_attribute in ["cluster_quality_label", "spike_depths"]:
        # The `index` parameter indicates whether the column is a "ragged array," i.e.
        # whether each row of this column is a vector with potentially different lengths
        # for each row.
        units_table.add_column(
            name=additional_attribute,
            description=ephys.CuratedClustering.Unit.heading.attributes[additional_attribute].comment,
            index=additional_attribute == "spike_depths",
        )

    clustering_query = (
        ephys.EphysRecording * ephys.ClusteringTask & session_key & paramset_record
    )

    for unit in tqdm(
        (ephys.CuratedClustering.Unit & clustering_query.proj()).fetch(as_dict=True),
        desc=f"creating units table for paramset {paramset_record['paramset_idx']}",
    ):

        probe_id, shank_num = (
            ephys.ProbeInsertion
            * ephys.CuratedClustering.Unit
            * probe.ProbeType.Electrode
            & unit
        ).fetch1("probe", "shank")

        waveform_mean = (
            ephys.WaveformSet.PeakWaveform() & clustering_query & unit
        ).fetch1("peak_electrode_waveform")

        units_table.add_row(
            id=unit["unit"],
            electrodes=[mapping[(probe_id, unit["electrode"])]],
            electrode_group=nwbfile.electrode_groups[
                f"probe{probe_id}_shank{shank_num}"
            ],
            cluster_quality_label=unit["cluster_quality_label"],
            spike_times=unit["spike_times"],
            spike_depths=unit["spike_depths"],
            waveform_mean=waveform_mean,
        )

    return units_table


def add_ephys_units_to_nwb(
    session_key: dict, nwbfile: pynwb.NWBFile, primary_clustering_paramset_idx: int = 0
):
    """
    Add spiking data to NWBFile.

    In NWB, spiking data is stored in a Units table. The primary Units table is
    stored at /units. The spiking data in /units is generally the data used in
    downstream analysis. Only a single Units table can be stored at /units. Other Units
    tables can be stored in a ProcessingModule at /processing/ecephys/. Any number of
    Units tables can be stored in this ProcessingModule as long as they have different
    names, and these Units tables can store intermediate processing steps or
    alternative curations.

    Use `primary_clustering_paramset_idx` to indicate which clustering is primary. All
    others will be stored in /processing/ecephys/.

    ephys.CuratedClustering.Unit::unit -> units.id
    ephys.CuratedClustering.Unit::spike_times -> units["spike_times"]
    ephys.CuratedClustering.Unit::spike_depths -> units["spike_depths"]
    ephys.CuratedClustering.Unit::cluster_quality_label -> units["cluster_quality_label"]

    ephys.WaveformSet.PeakWaveform::peak_electrode_waveform -> units["waveform_mean"]

    Parameters
    ----------
    session_key: dict
    nwbfile: pynwb.NWBFile
    primary_clustering_paramset_idx: int, optional
    """

    if not ephys.ClusteringTask & session_key:
        warnings.warn(f'No unit data exists for session:{session_key}')
        return

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    for paramset_record in (
        ephys.ClusteringParamSet & ephys.CuratedClustering & session_key
    ).fetch("paramset_idx", "clustering_method", "paramset_desc", as_dict=True):
        if paramset_record["paramset_idx"] == primary_clustering_paramset_idx:
            units_table = create_units_table(
                session_key,
                nwbfile,
                paramset_record,
                desc=paramset_record["paramset_desc"],
            )
            nwbfile.units = units_table
        else:
            name = f"units_{paramset_record['clustering_method']}"
            units_table = create_units_table(
                session_key,
                nwbfile,
                paramset_record,
                name=name,
                desc=paramset_record["paramset_desc"],
            )
            ecephys_module = get_module(nwbfile, "ecephys")
            ecephys_module.add(units_table)


def get_electrodes_mapping(electrodes):
    """
    Create a mapping from the probe and electrode id to the row number of the electrodes
    table. This is used in the construction of the DynamicTableRegion that indicates what rows of the electrodes
    table correspond to the data in an ElectricalSeries.

    Parameters
    ----------
    electrodes: hdmf.common.table.DynamicTable

    Returns
    -------
    dict

    """
    return {
        (electrodes["group"][idx].device.name, electrodes["id_in_probe"][idx],): idx
        for idx in range(len(electrodes))
    }


def gains_helper(gains):
    """
    This handles three different cases for gains:
    1. gains are all 1. In this case, return conversion=1e-6, which applies to all
    channels and converts from microvolts to volts.
    2. Gains are all equal, but not 1. In this case, multiply this by 1e-6 to apply this
    gain to all channels and convert units to volts.
    3. Gains are different for different channels. In this case use the
    `channel_conversion` field in addition to the `conversion` field so that each
    channel can be converted to volts using its own individual gain.

    Parameters
    ----------
    gains: np.ndarray

    Returns
    -------
    dict
        conversion : float
        channel_conversion : np.ndarray

    """
    if all(x == 1 for x in gains):
        return dict(conversion=1e-6, channel_conversion=None)
    if all(x == gains[0] for x in gains):
        return dict(conversion=1e-6 * gains[0], channel_conversion=None)
    return dict(conversion=1e-6, channel_conversion=gains)


def add_ephys_recording_to_nwb(
    session_key: dict,
    ephys_root_data_dir: str,
    nwbfile: pynwb.NWBFile,
    end_frame: int = None,
):
    """
    Read voltage data directly from source files and iteratively transfer them to the NWB file. Automatically
    applies lossless compression to the data, so the final file might be smaller than the original, without
    data loss. Currently supports Neuropixels data acquired with SpikeGLX or Open Ephys, and relies on SpikeInterface to read the data.

    source data -> acquisition["ElectricalSeries"]

    Parameters
    ----------
    session_key: dict
    ephys_root_data_dir: str
    nwbfile: NWBFile
    end_frame: int, optional
        Used for small test conversions
    """

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    mapping = get_electrodes_mapping(nwbfile.electrodes)

    for ephys_recording_record in (ephys.EphysRecording & session_key).fetch(
        as_dict=True
    ):
        probe_id = (ephys.ProbeInsertion() & ephys_recording_record).fetch1("probe")

        relative_path = (
            ephys.EphysRecording.EphysFile & ephys_recording_record
        ).fetch1("file_path")
        relative_path = relative_path.replace("\\","/")
        file_path = find_full_path(ephys_root_data_dir, relative_path)

        if ephys_recording_record["acq_software"] == "SpikeGLX":
            extractor = extractors.read_spikeglx(os.path.split(file_path)[0], "imec.ap")
        elif ephys_recording_record["acq_software"] == "OpenEphys":
            extractor = extractors.read_openephys(file_path, stream_id="0")
        else:
            raise ValueError(
                f"unsupported acq_software type: {ephys_recording_record['acq_software']}"
            )

        conversion_kwargs = gains_helper(extractor.get_channel_gains())

        if end_frame is not None:
            extractor = extractor.frame_slice(0, end_frame)

        recording_channels_by_id = (
            probe.ElectrodeConfig.Electrode() & ephys_recording_record
        ).fetch("electrode")

        nwbfile.add_acquisition(
            pynwb.ecephys.ElectricalSeries(
                name=f"ElectricalSeries{ephys_recording_record['insertion_number']}",
                description=str(ephys_recording_record),
                data=SpikeInterfaceRecordingDataChunkIterator(extractor),
                rate=ephys_recording_record["sampling_rate"],
                starting_time=(
                    ephys_recording_record["recording_datetime"]
                    - ephys_recording_record["session_datetime"]
                ).total_seconds(),
                electrodes=nwbfile.create_electrode_table_region(
                    region=[mapping[(probe_id, x)] for x in recording_channels_by_id],
                    name="electrodes",
                    description="recorded electrodes",
                ),
                **conversion_kwargs
            )
        )


def add_ephys_lfp_from_dj_to_nwb(session_key: dict, nwbfile: pynwb.NWBFile):
    """
    Read LFP data from the data in element-aray-ephys

    ephys.LFP.Electrode::lfp -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].data
    ephys.LFP::lfp_time_stamps -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].timestamps

    Parameters
    ----------
    session_key: dict
    nwbfile: NWBFile
    """

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    ecephys_module = get_module(
        nwbfile, name="ecephys", description="preprocessed ephys data"
    )

    nwb_lfp = pynwb.ecephys.LFP(name="LFP")
    ecephys_module.add(nwb_lfp)

    mapping = get_electrodes_mapping(nwbfile.electrodes)

    for lfp_record in (ephys.LFP & session_key).fetch(as_dict=True):
        probe_id = (ephys.ProbeInsertion & lfp_record).fetch1("probe")

        lfp_electrodes_query = ephys.LFP.Electrode & lfp_record
        lfp_data = LFPDataChunkIterator(lfp_electrodes_query)

        nwb_lfp.create_electrical_series(
            name=f"ElectricalSeries{lfp_record['insertion_number']}",
            description=f"LFP from probe {probe_id}",
            data=H5DataIO(lfp_data, compression=True),
            timestamps=lfp_record["lfp_time_stamps"],
            electrodes=nwbfile.create_electrode_table_region(
                name="electrodes",
                description="electrodes used for LFP",
                region=[
                    mapping[(probe_id, x)]
                    for x in lfp_electrodes_query.fetch("electrode")
                ],
            ),
        )


def add_ephys_lfp_from_source_to_nwb(
    session_key: dict, ephys_root_data_dir, nwbfile: pynwb.NWBFile, end_frame=None):
    """
    Read the LFP data directly from the source file. Currently, only works for SpikeGLX data.

    ephys.EphysRecording::recording_datetime -> acquisition

    Parameters
    ----------
    session_key: dict
    nwbfile: pynwb.NWBFile
    end_frame: int, optional
        use for small test conversions

    """
    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    mapping = get_electrodes_mapping(nwbfile.electrodes)

    ecephys_module = get_module(
        nwbfile, name="ecephys", description="preprocessed ephys data"
    )

    lfp = pynwb.ecephys.LFP()
    ecephys_module.add(lfp)

    for ephys_recording_record in (ephys.EphysRecording & session_key).fetch(
        as_dict=True
    ):
        probe_id = (ephys.ProbeInsertion() & ephys_recording_record).fetch1("probe")

        relative_path = (
            ephys.EphysRecording.EphysFile & ephys_recording_record
        ).fetch1("file_path")
        relative_path = relative_path.replace("\\","/")
        file_path = find_full_path(ephys_root_data_dir, relative_path)

        if ephys_recording_record["acq_software"] == "SpikeGLX":
            extractor = extractors.read_spikeglx(os.path.split(file_path)[0], "imec.lf")
        else:
            raise ValueError(
                f"unsupported acq_software type: {ephys_recording_record['acq_software']}"
            )

        if end_frame is not None:
            extractor = extractor.frame_slice(0, end_frame)

        recording_channels_by_id = (
            probe.ElectrodeConfig.Electrode() & ephys_recording_record
        ).fetch("electrode")

        conversion_kwargs = gains_helper(extractor.get_channel_gains())

        lfp.add_electrical_series(
            pynwb.ecephys.ElectricalSeries(
                name=f"ElectricalSeries{ephys_recording_record['insertion_number']}",
                description=f"LFP from probe {probe_id}",
                data=SpikeInterfaceRecordingDataChunkIterator(extractor),
                rate=extractor.get_sampling_frequency(),
                starting_time=(
                    ephys_recording_record["recording_datetime"]
                    - ephys_recording_record["session_datetime"]
                ).total_seconds(),
                electrodes=nwbfile.create_electrode_table_region(
                    region=[mapping[(probe_id, x)] for x in recording_channels_by_id],
                    name="electrodes",
                    description="recorded electrodes",
                ),
                **conversion_kwargs
            )
        )


def ecephys_session_to_nwb(
    session_key,
    raw=True,
    spikes=True,
    lfp="source",
    end_frame=None,
    lab_key=None,
    project_key=None,
    protocol_key=None,
    nwbfile_kwargs=None,
):
    """
    Main function for converting ephys data to NWB

    Parameters
    ----------
    session_key: dict
    raw: bool
        Whether to include the raw data from source. SpikeGLX and OpenEphys are supported
    spikes: bool
        Whether to include CuratedClustering
    lfp:
        "dj" - read LFP data from ephys.LFP
        "source" - read LFP data from source (SpikeGLX supported)
        False - do not convert LFP
    end_frame: int, optional
        Used to create small test conversions where large datasets are truncated.
    lab_key, project_key, and protocol_key: dictionaries used to look up optional additional metadata
    nwbfile_kwargs: dict, optional
        - If element-session is not being used, this argument is required and must be a dictionary containing
          'session_description' (str), 'identifier' (str), and 'session_start_time' (datetime),
          the minimal data for instantiating an NWBFile object.

        - If element-session is being used, this argument can optionally be used to add over overwrite NWBFile fields.
    """

    session_to_nwb = getattr(ephys._linking_module, 'session_to_nwb', False)

    if session_to_nwb:
        nwbfile = session_to_nwb(
            session_key,
            lab_key=lab_key,
            project_key=project_key,
            protocol_key=protocol_key,
            additional_nwbfile_kwargs=nwbfile_kwargs,
        )
    else:
        nwbfile = pynwb.NWBFile(**nwbfile_kwargs)

    ephys_root_data_dir = ephys.get_ephys_root_data_dir()

    if raw:
        add_ephys_recording_to_nwb(session_key, ephys_root_data_dir=ephys_root_data_dir,
                                   nwbfile=nwbfile, end_frame=end_frame)

    if spikes:
        add_ephys_units_to_nwb(session_key, nwbfile)

    if lfp == "dj":
        add_ephys_lfp_from_dj_to_nwb(session_key, nwbfile)

    if lfp == "source":
        add_ephys_lfp_from_source_to_nwb(session_key, ephys_root_data_dir=ephys_root_data_dir,
                                         nwbfile=nwbfile, end_frame=end_frame)

    return nwbfile


def write_nwb(nwbfile, fname, check_read=True):
    """
    Export NWBFile

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    fname: str Absolute path including `*.nwb` extension.
    check_read: bool
        If True, PyNWB will try to read the produced NWB file and ensure that it can be
        read.
    """
    with pynwb.NWBHDF5IO(fname, "w") as io:
        io.write(nwbfile)

    if check_read:
        with pynwb.NWBHDF5IO(fname, "r") as io:
            io.read()
