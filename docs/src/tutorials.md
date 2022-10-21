# Tutorials

## Installation

Installation of the Element requires an integrated development environment and database. Instructions to setup each of the components can be found on the [User Instructions](datajoint.com/docs/elements/user-instructions) page. These instructions use the example [workflow for Element Array Electrophysiology](https://github.com/datajoint/workflow-array-ephys), which can be modified for a user's specific experimental requirements.

## YouTube

Our [Element Array Electrophysiology tutorial](https://www.youtube.com/watch?v=KQlGYOBq7ow) covers a detailed explanation of the workflow as well as core concepts of the electrophysiology data pipeline.

[![YouTube tutorial](https://img.youtube.com/vi/KQlGYOBq7ow/0.jpg)](https://www.youtube.com/watch?v=KQlGYOBq7ow)


## Notebooks

Each of the [notebooks](https://github.com/datajoint/workflow-array-ephys/tree/main/notebooks) in the workflow steps through ways to interact with the Element itself. To try out Elements notebooks in an online Jupyter environment with access to example data, visit [CodeBook](https://codebook.datajoint.io/).

- [00-DataDownload](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/00-data-download-optional.ipynb) highlights how to use DataJoint tools to download a sample dataset for trying out the Element.
- [01-Configure](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/01-configure.ipynb) helps configure your local DataJoint installation to point to the correct database.
- [02-WorkflowStructure](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/02-workflow-structure-optional.ipynb) demonstrates the table architecture of the Element and key DataJoint basics for interacting with these tables.
- [03-Process](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/03-process.ipynb) steps through adding data to these tables and launching key analysis features of the Element.
- [04-Automate](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/04-automate-optional.ipynb) highlights the same steps as above, but utilizing all built-in automation tools.
- [05-Exploration](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/05-explore.ipynb) demonstrates how to filter and fetch data from the Element to generate figures.
- [06-Drop](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/06-drop-optional.ipynb) provides the steps for dropping all the tables to start fresh.
- [07-Analysis](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/07-downstream-analysis.ipynb) demonstrates how to perform downstream analysis on electrophysiology data such as event-alignment.
- [08-ElectrodeLocalization](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/08-electrode-localization.ipynb) provides the steps to perform stereotaxic electrode localization using the Allen Brain Atlases.

## Data Export to Neurodata Without Borders (NWB)

The [final notebook](https://github.com/datajoint/workflow-array-ephys/blob/main/notebooks/09-NWB-export.ipynb) in the Element Array Electrophysiology tutorials provides a walk-through to export data generated through the Array Electrophysiology workflow to NWB format and upload to DANDI archive. Details on exporting electrophysiology files through NWB are provided below.

+ The export/nwb/nwb.py module maps from the `element-array-ephys` data structure to NWB. The main function is `ecephys_session_to_nwb`, which contains flags to control calling the following functions, which can be called independently:

    + `session.export.nwb.session_to_nwb`: Gathers session-level metadata

    + `add_electrodes_to_nwb`: Add electrodes table to NWBFile. This is needed for any `ElectricalSeries`, including raw source data and LFP.

    + `ephys.InsertionLocation` -> `ElectrodeGroup.location`

    + `probe.Probe`::`probe` -> `device.name` `probe.Probe`::`probe_comment` -> `device.description` `probe.Probe`::`probe_type` -> `device.manufacturer`

    + `probe.ProbeType.Electrode`::`electrode` -> `electrodes["id_in_probe"] probe.ProbeType.Electrode`::`y_coord` -> `electrodes["rel_y"] probe.ProbeType.Electrode`::`x_coord` -> `electrodes["rel_x"] probe.ProbeType.Electrode`::`shank` -> `electrodes["shank"]` `probe.ProbeType.Electrode`::`shank_col` -> `electrodes["shank_col"]` `probe.ProbeType.Electrode`::`shank_row` -> `electrodes["shank_row"]`

    + `add_ephys_recording_to_nwb`: Read voltage data directly from source files and iteratively transfer them to the NWB file. Automatically applies lossless compression to the data, so the final file might be smaller than the original, but there is no data loss. Currently supports Neuropixels data acquired with SpikeGLX or Open Ephys, and relies on SpikeInterface to read the data.

    + source data -> `acquisition["ElectricalSeries"]`

    + `add_ephys_units_to_nwb`: Add spiking data from CuratedClustering to NWBFile.

    + `ephys.CuratedClustering.Unit`::`unit` -> `units.id` `ephys.CuratedClustering.Unit`::`spike_times` -> `units["spike_times"]` `ephys.CuratedClustering.Unit`::`spike_depths` -> `units["spike_depths"]` `ephys.CuratedClustering.Unit`::`cluster_quality_label` -> `units["cluster_quality_label"]`

    + `ephys.WaveformSet.PeakWaveform`::`peak_electrode_waveform` -> `units["waveform_mean"]`

    + `add_ephys_lfp_from_dj_to_nwb`: Read LFP data from the data in element-array-ephys and convert to NWB.

    + `ephys.LFP.Electrode`::`lfp` -> `processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].data` `ephys.LFP`::`lfp_time_stamps` -> `processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].timestamps`

    + `add_ephys_lfp_from_source_to_nwb`: Read the LFP data directly from the source file. Currently, only works for SpikeGLX data. Can be used instead of `add_ephys_lfp_from_dj_to_nwb`.

    + source data -> `processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].data` source data -> `processing["ecephys"].lfp.electrical_series["ElectricalSeries{insertion_number}"].timestamps`
