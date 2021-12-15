from hdmf.backends.hdf5 import H5DataIO
from nwb_conversion_tools.utils.genericdatachunkiterator import GenericDataChunkIterator
from nwb_conversion_tools.utils.recordingextractordatachunkiterator import (
    RecordingExtractorDataChunkIterator,
)
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.misc import Units
from spikeextractors import (
    SpikeGLXRecordingExtractor,
    SubRecordingExtractor,
    OpenEphysNPIXRecordingExtractor,
)
from tqdm import tqdm

from element_session import session


# from element_data_loader.utils import find_full_path


class LFPDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator for LFP data that pulls data one channel at a time."""

    def __init__(
        self,
        lfp_electrodes_query,
        buffer_length: int = None,
        chunk_length: int = 10000,
    ):
        self.lfp_electrodes_query = lfp_electrodes_query
        self.electrodes = self.lfp_electrodes_query.fetch("electrode")

        first_record = (
            self.lfp_electrodes_query & dict(electrode=self.electrodes[0])
        ).fetch(as_dict=True)[0]

        self.n_channels = len(lfp_electrodes_query)
        self.n_tt = len(first_record["lfp"])
        self._dtype = first_record["lfp"].dtype

        chunk_shape = (chunk_length, 1)
        if buffer_length is not None:
            buffer_shape = (buffer_length, 1)
        else:
            buffer_shape = (len(first_record["lfp"]), 1)
        super().__init__(buffer_shape=buffer_shape, chunk_shape=chunk_shape)

    def _get_data(self, selection):

        electrode = self.electrodes[selection[1]][0]
        return (self.lfp_electrodes_query & dict(electrode=electrode)).fetch1("lfp")[
            selection[0], np.newaxis
        ]

    def _get_dtype(self):
        return self._dtype

    def _get_maxshape(self):
        return self.n_tt, self.n_channels


def check_module(nwbfile, name: str, description: str = None):
    """
    Check if processing module exists. If not, create it. Then return module.
    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)
    Returns
    -------
    pynwb.module
    """
    assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
    if name in nwbfile.processing:
        return nwbfile.processing[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)


def get_ephys_root_data_dir():
    root_data_dirs = dj.config.get("custom", {}).get("ephys_root_data_dir", None)
    return root_data_dirs


def add_electrodes_to_nwb(session_key: dict, nwbfile: NWBFile):

    electrodes_query = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode

    session_electrodes = (
        probe.ElectrodeConfig.Electrode & (ephys.EphysRecording & session_key)
    ).fetch()

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

    session_probes = (ephys.ProbeInsertion * probe.Probe & session_key).fetch(
        as_dict=True
    )
    for this_probe in session_probes:

        insertion_record = (ephys.InsertionLocation & this_probe).fetch()
        if False:  # insertion_record:
            insert_location = insertion_record.fetch(*non_primary_keys, as_dict=True)
            insert_location = json.dumps(insert_location)
        else:
            insert_location = "unknown"

        device = nwbfile.create_device(
            name=this_probe["probe"],
            description=this_probe.get("probe_comment", None),
            manufacturer=this_probe.get("probe_type", None),
        )
        shank_ids = np.unique((probe.ProbeType.Electrode & this_probe).fetch("shank"))
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
                    x=np.nan,  # to do: populate these values once the CCF element is ready
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
    session_key,
    nwbfile,
    units_query,
    paramset_record,
    name="units",
    desc="data on spiking units",
):

    # electrode id mapping
    electrode_id_mapping = {
        nwbfile.electrodes.id.data[x]: x for x in range(len(nwbfile.electrodes.id))
    }

    units_table = Units(name=name, description=desc)
    for additional_attribute in ["cluster_quality_label", "spike_depths"]:
        units_table.add_column(
            name=units_query.heading.attributes[additional_attribute].name,
            description=units_query.heading.attributes[additional_attribute].comment,
            index=True if additional_attribute == "spike_depths" else False,
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
            electrodes=[electrode_id_mapping[unit["electrode"]]],
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
    session_key: dict, nwbfile: NWBFile, primary_clustering_paramset_idx: int = 0
):

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
                units_query,
                paramset_record,
                name=name,
                desc=paramset_record["paramset_desc"],
            )
            ecephys_module = check_module(nwbfile, "ecephys")
            ecephys_module.add(units_table)


def add_ephys_recording_to_nwb(
    session_key: dict, nwbfile: NWBFile, end_frame: int = None
):

    if not ephys.EphysRecording & session_key:
        return

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    mapping = {
        (
            nwbfile.electrodes["group"][idx].device.name,
            nwbfile.electrodes["id_in_probe"][idx],
        ): idx
        for idx in range(len(nwbfile.electrodes))
    }

    for ephys_recording_record in (ephys.EphysRecording & session_key).fetch(
        as_dict=True
    ):
        probe_id = (ephys.ProbeInsertion() & ephys_recording_record).fetch1("probe")

        # relative_path = (ephys.EphysRecording.EphysFile & ephys_recording_record).fetch1("file_path")
        # file_path = find_full_path(get_ephys_root_data_dir(), relative_path)
        file_path = "../inbox/subject5/session1/probe_1/npx_g0_t0.imec.ap.bin"

        if ephys_recording_record["acq_software"] == "SpikeGLX":
            extractor = SpikeGLXRecordingExtractor(file_path=file_path)
        elif ephys_recording_record["acq_software"] == "OpenEphys":
            extractor = OpenEphysNPIXRecordingExtractor(file_path=file_path)
        channel_conversions = extractor.get_channel_gains()
        # to do: channel conversions for OpenEphys

        if end_frame is not None:
            extractor = SubRecordingExtractor(extractor, end_frame=end_frame)

        recording_channels_by_id = (
            probe.ElectrodeConfig.Electrode()
            & ephys.EphysRecording()
            & session_key
            & dict(insertion_number=1)
        ).fetch("electrode")

        nwbfile.add_acquisition(
            ElectricalSeries(  # to do: add conversion
                name=f"ElectricalSeries{ephys_recording_record['insertion_number']}",
                description=str(ephys_recording_record),
                data=H5DataIO(
                    RecordingExtractorDataChunkIterator(extractor), compression=True
                ),
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
                conversion=1e-6,
                channel_conversion=channel_conversions,
            )
        )


def add_ephys_lfp_to_nwb(session_key: dict, nwbfile: NWBFile):

    if not ephys.LFP & session_key:
        return

    if nwbfile.electrodes is None:
        add_electrodes_to_nwb(session_key, nwbfile)

    ecephys_module = check_module(
        nwbfile, name="ecephys", description="preprocessed ephys data"
    )

    nwb_lfp = LFP(name="LFP")
    ecephys_module.add(nwb_lfp)

    mapping = {
        (
            nwbfile.electrodes["group"][idx].device.name,
            nwbfile.electrodes["id_in_probe"][idx],
        ): idx
        for idx in range(len(nwbfile.electrodes))
    }

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

def session_to_nwb(session_key, subject_id=None, raw=True, spikes=True, lfp=True, end_frame=None):

    nwbfile = session.export.nwb.session_to_nwb(session_key, subject_id=subject_id)

    if raw:
        add_ephys_recording_to_nwb(session_key, nwbfile, end_frame=end_frame)

    if spikes:
        add_ephys_units_to_nwb(session_key, nwbfile)

    if lfp:
        add_ephys_lfp_to_nwb(session_key, nwbfile)

    return nwbfile

def write_nwb(nwbfile, fname, check_read=True):
    with NWBHDF5IO(fname, 'w') as io:
        io.write(nwbfile)

    if check_read:
        with NWBHDF5IO(fname, 'r') as io:
            io.read()
