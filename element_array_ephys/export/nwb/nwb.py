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

from ... import probe, ephys_acute, ephys_chronic, ephys_no_curation

assert probe.schema.is_activated(), 'probe not yet activated'

for ephys in (ephys_acute, ephys_chronic, ephys_no_curation):
    if ephys.schema.is_activated():
        break
else:
    raise AssertionError('ephys not yet activated')


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
        lfp_electrodes_query: element_array_ephys.ephys_no_curation.LFP
        chunk_length: int, optional
            Chunks are blocks of disk space where data are stored contiguously and compressed
        """
        self.lfp_electrodes_query = lfp_electrodes_query
        self.electrodes = self.lfp_electrodes_query.fetch("electrode")

        first_record = (
            self.lfp_electrodes_query & dict(electrode=self.electrodes[0])
        ).fetch(as_dict=True)[0]

        self.n_channels = len(lfp_electrodes_query)
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
        insertion_record = (ephys.InsertionLocation & this_probe).fetch1()
        if insertion_record:
            insert_location = json.dumps(
                {
                    k: v
                    for k, v in insertion_record.items()
                    if k not in ephys.InsertionLocation.primary_key
                },
                cls=DecimalEncoder,
            )
        else:
            insert_location = "unknown"

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
    units_query,
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
    units_query: ephys.CuratedClustering.Unit
    paramset_record: int
    name: str, optional
        default="units"
    desc: str, optional
        default="data on spiking units"
    """

    # electrode id mapping
    mapping = get_electrodes_mapping(nwbfile.electrodes)

    units_table = pynwb.misc.Units(name=name, description=desc)
    for additional_attribute in ["cluster_quality_label", "spike_depths"]:
        units_table.add_column(
            name=units_query.heading.attributes[additional_attribute].name,
            description=units_query.heading.attributes[additional_attribute].comment,
            index=additional_attribute == "spike_depths",
        )

    clustering_query = (
        ephys.EphysRecording * ephys.ClusteringTask & session_key & paramset_record
    )

    for unit in tqdm(
        (clustering_query @ ephys.CuratedClustering.Unit).fetch(as_dict=True),
        desc=f"creating units table for paramset {paramset_record['paramset_idx']}",
    ):

        probe_id, shank_num = (
            probe.ProbeType.Electrode
            * ephys.ProbeInsertion
            * ephys.CuratedClustering.Unit
            & {"unit": unit["unit"], "insertion_number": unit["insertion_number"]}
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
        return

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    # add additional columns to the units table
    units_query = ephys.CuratedClustering.Unit() & session_key

    for paramset_record in (
        ephys.ClusteringParamSet & ephys.CuratedClustering & session_key
    ).fetch("paramset_idx", "clustering_method", "paramset_desc", as_dict=True):
        if paramset_record["paramset_idx"] == primary_clustering_paramset_idx:
            units_table = create_units_table(
                session_key,
                nwbfile,
                units_query,
                paramset_record,
                desc=paramset_record["paramset_desc"],
            )
            nwbfile.units = units_table
        else:
            name = f"units_{paramset_record['clustering_method']}"
            units_table = create_units_table(
                session_key,
                nwbfile,
                units_query,
                paramset_record,
                name=name,
                desc=paramset_record["paramset_desc"],
            )
            ecephys_module = get_module(nwbfile, "ecephys")
            ecephys_module.add(units_table)


def get_electrodes_mapping(electrodes):
    """
    Create a mapping from the group (shank) and electrode id within that group to the row number of the electrodes
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
    if all(x == 1 for x in gains):
        return dict(conversion=1e-6, channel_conversion=None)
    if all(x == gains[0] for x in gains):
        return dict(conversion=1e-6 * gains[0], channel_conversion=None)
    return dict(conversion=1e-6, channel_conversion=gains)


def add_ephys_recording_to_nwb(
    session_key: dict, ephys_root_data_dir,
    nwbfile: pynwb.NWBFile, end_frame: int = None):
    """Read voltage data directly from source files and iteratively transfer them to the NWB file. Automatically
    applies lossless compression to the data, so the final file might be smaller than the original, without
    data loss. Currently supports neuropixel and openephys, and relies on SpikeInterface to read the data.

    source data -> acquisition["ElectricalSeries"]

    Parameters
    ----------
    session_key: dict
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
                description=str(ephys_recording_record),
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
