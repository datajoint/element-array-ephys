# Exporting data to NWB

To use the export functionality, install the Element with the `nwb` option as follows:

```console
pip install element-array-ephys[nwb]
```

The `export/nwb/nwb.py` module maps from the element-array-ephys data structure to NWB.
The main function is `ecephys_session_to_nwb`, which contains flags to control calling
the following functions, which can be called independently:

1. `session.export.nwb.session_to_nwb`: Gathers session-level metadata

2. `add_electrodes_to_nwb`: Add electrodes table to NWBFile. This is needed for any
   ElectricalSeries, including raw source data and LFP.

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

3. `add_ephys_recording_to_nwb`: Read voltage data directly from source files and
   iteratively transfer them to the NWB file. Automatically applies lossless compression
   to the data, so the final file might be smaller than the original, but there is no
   data loss. Currently supports Neuropixels data acquired with SpikeGLX or Open Ephys,
   and relies on SpikeInterface to read the data.

    source data -> acquisition["ElectricalSeries"]

4. `add_ephys_units_to_nwb`: Add spiking data from CuratedClustering to NWBFile.

    ephys.CuratedClustering.Unit::unit -> units.id
    ephys.CuratedClustering.Unit::spike_times -> units["spike_times"]
    ephys.CuratedClustering.Unit::spike_depths -> units["spike_depths"]
    ephys.CuratedClustering.Unit::cluster_quality_label -> units["cluster_quality_label"]

    ephys.WaveformSet.PeakWaveform::peak_electrode_waveform -> units["waveform_mean"]

5. `add_ephys_lfp_from_dj_to_nwb`: Read LFP data from the data in element-array-ephys
   and convert to NWB.

    ephys.LFP.Electrode::lfp -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].data
    ephys.LFP::lfp_time_stamps -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].timestamps

6. `add_ephys_lfp_from_source_to_nwb`: Read the LFP data directly from the source file.
   Currently, only works for SpikeGLX data. Can be used instead of
   `add_ephys_lfp_from_dj_to_nwb`.

    source data -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].data
    source data -> processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].timestamps
