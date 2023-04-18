# Concepts

## Acquisition Tools for Electrophysiology

### Neuropixels Probes

Neuropixels probes were developed by a collaboration between HHMI Janelia, industry
partners, and others[^1]. Since their initial release in October
2018, 300 labs have ordered 1200 probes. Since the rollout of Neuropixels 2.0 in October
2020, IMEC has been shipping 100+ probes monthly (correspondence with Tim Harris).

Neuropixels probes offer 960 electrode sites along a 10mm long shank, with 384
recordable channels per probe that can record hundreds of units spanning multiple brain
regions (Neuropixels 2.0 version is a 4-shank probe with 1280 electrode sites per
shank). Such large recording capacity has offered tremendous opportunities for the field
of neurophysiology research, yet this is accompanied by an equally great challenge in
terms of data and computation management.

[^1]: 
    Jun, J., Steinmetz, N., Siegle, J. et al. Fully integrated silicon probes for
    high-density recording of neural activity. *Nature* 551, 232–236 (2017).
    [https://doi.org/10.1038/nature24636](https://doi.org/10.1038/nature24636).

### Data Acquisition Tools

Some commonly used acquisition tools and systems by the neuroscience research community
include:

+ [Neuropixels probes](https://www.neuropixels.org)
+ Tetrodes
+ [SpikeGLX](http://billkarsh.github.io/SpikeGLX/)
+ [OpenEphys](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html)
+ [Neuralynx](https://neuralynx.com/)
+ [Axona](http://www.axona.com/)
+ ...

### Data Preprocessing Tools

The preprocessing pipeline includes bandpass filtering for LFP extraction, bandpass
filtering for spike sorting, spike sorting, manual curation of the spike sorting
results, and calculation of quality control metrics. In trial-based experiments, the
spike trains are aligned and separated into trials. Standard processing may include PSTH
computation aligned to trial onset or other events, and often grouped by different trial
types. Neuroscience groups have traditionally developed custom home-made toolchains.

In recent years, several leaders have been emerging as de facto standards with
significant community uptake:

+ [Kilosort](https://github.com/MouseLand/Kilosort)
+ [pyKilosort](https://github.com/MouseLand/pykilosort)
+ [JRClust](https://github.com/JaneliaSciComp/JRCLUST)
+ [KlustaKwik](https://klusta.readthedocs.io/en/latest/)
+ [Mountainsort](https://github.com/flatironinstitute/mountainsort)
+ [spikeinterface (wrapper)](https://github.com/SpikeInterface)
+ [spyking-circus](https://github.com/spyking-circus/spyking-circus)
+ [spikeforest](https://spikeforest.flatironinstitute.org/)
+ ...

Kilosort provides most automation and has gained significant popularity, being adopted
as one of the key spike sorting methods in the majority of the teams/collaborations we
have worked with. As part of our Year-1 NIH U24 effort, we provide support for data
ingestion of spike sorting results from Kilosort. Further effort will be devoted for the
ingestion support of other spike sorting methods. On this end, a framework for unifying
existing spike sorting methods, named
[SpikeInterface](https://github.com/SpikeInterface/spikeinterface), has been developed
by Alessio Buccino, et al. SpikeInterface provides a convenient Python-based wrapper to
invoke, extract, compare spike sorting results from different sorting algorithms.

## Key Partnerships

Over the past few years, several labs have developed DataJoint-based data management and
processing pipelines for Neuropixels probes. Our team collaborated with several of them
during their projects. Additionally, we interviewed these teams to understand their
experimental workflow, pipeline design, associated tools, and interfaces. These teams
include:

+ [International Brain Lab](https://internationalbrainlab.org):
  [https://github.com/int-brain-lab/IBL-pipeline](https://github.com/int-brain-lab/IBL-pipeline)

+ [Mesoscale Activity Project (HHMI Janelia)](https://github.com/mesoscale-activity-map):
  [https://github.com/mesoscale-activity-map/map-ephys](https://github.com/mesoscale-activity-map/map-ephys)

+ Moser Group (Norwegian University of Science and Technology): see 
  [pipeline design](https://moser-pipelines.readthedocs.io/en/latest/ephys/overview.html)

+ Andreas Tolias Lab (Baylor College of Medicine)

+ BrainCoGs (Princeton Neuroscience Institute)

+ Brody Lab (Princeton University)

## Element Architecture

Each of the DataJoint Elements creates a set of tables for common neuroscience data
modalities to organize, preprocess, and analyze data. Each node in the following diagram
is a table within the Element or a table connected to the Element.

### `ephys_acute` module

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_acute.svg)

### `ephys_chronic` module

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_chronic.svg)

### `ephys_precluster` module

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_precluster.svg)

### `ephys_no_curation` module

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_no_curation.svg)

### `subject` schema ([API docs](https://datajoint.com/docs/elements/element-animal/api/element_animal/subject))

Although not required, most choose to connect the `Session` table to a `Subject` table.

| Table | Description |
| --- | --- |
| Subject | A table containing basic information of the research subject. |

### `session` schema ([API docs](https://datajoint.com/docs/elements/element-session/api/element_session/session_with_datetime))

| Table | Description |
| --- | --- |
| Session | A table for unique experimental session identifiers. |

### `probe` schema ([API docs](../api/element_array_ephys/probe))

Tables related to the Neuropixels probe and electrode configuration.

| Table | Description |
| --- | --- |
| ProbeType | A lookup table specifying the type of Neuropixels probe (e.g. "neuropixels 1.0", "neuropixels 2.0 single-shank"). |
| ProbeType.Electrode | A table containing electrodes and their properties for a particular probe type. |
| Probe | A record of an actual physical probe. |
| ElectrodeConfig | A record of a particular electrode configuration to be used for ephys recording. |
| ElectrodeConfig.Electrode | A record of electrodes out of those in `ProbeType.Electrode` that are used for recording. |

### `ephys` schema ([API docs](../api/element_array_ephys/ephys))

Tables related to information about physiological recordings and automatic ingestion of
spike sorting results.

| Table | Description |
| --- | --- |
| ProbeInsertion | A record of surgical insertions of a probe in the brain. |
| EphysRecording | A table with metadata about each electrophysiogical recording. |
| Clustering | A table with clustering data for spike sorting extracellular electrophysiology data. |
| Curation | A table to declare optional manual curation of spike sorting results. |
| CuratedClustering | A table with metadata for sorted data generated after each curation. |
| CuratedClustering.Unit | A part table containing single unit information after spike sorting and optional curation. |
| WaveformSet | A table containing spike waveforms for single units. |

### `ephys_report` schema ([API docs](../api/element_array_ephys/ephys_report))

Tables for storing probe or unit-level visualization results.

| Table | Description |
| --- | --- |
| ProbeLevelReport | A table to store drift map figures generated from each recording probe. |
| UnitLevelReport | A table to store figures (waveforms, autocorrelogram, peak waveform + neighbors) generated for each unit. |
| QualityMetricCutoffs | A table to store cut-off values for cluster quality metrics. |
| QualityMetricSet | A manual table to match a set of cluster quality metric values with desired cut-offs. |
| QualityMetricReport | A table to store quality metric figures. |

## Element Development

Through our interviews and direct collaboration on the precursor projects, we identified
the common motifs to create the 
[Array Electrophysiology Element](https://github.com/datajoint/element-array-ephys).

Major features of the Array Electrophysiology Element include:

+ Pipeline architecture detailed by:

    + Probe, electrode configuration compatible with Neuropixels probes and
      generalizable to other types of probes (e.g. tetrodes) - supporting both `chronic`
      and `acute` probe insertion modes.

    + Probe-insertion, ephys-recordings, LFP extraction, clusterings, curations, sorted
      units and the associated data (e.g. spikes, waveforms, etc.).
    
    + Store/track/manage different curations of the spike sorting results - supporting
      both curated clustering and kilosort triggered clustering (i.e., `no_curation`).

+ Ingestion support for data acquired with SpikeGLX and OpenEphys acquisition systems. 
+ Ingestion support for spike sorting outputs from Kilosort.
+ Triggering support for workflow integrated Kilosort processing.
+ Sample data and complete test suite for quality assurance.

## Data Export and Publishing

Element Array Electrophysiology supports exporting of all data into standard Neurodata
Without Borders (NWB) files. This makes it easy to share files with collaborators and
publish results on [DANDI Archive](https://dandiarchive.org/).
[NWB](https://www.nwb.org/), as an organization, is dedicated to standardizing data
formats and maximizing interoperability across tools for neurophysiology. For more
information on uploading NWB files to DANDI within the DataJoint Elements ecosystem see
the corresponding notebook on the [tutorials page](./tutorials/index.md). 

To use the export functionality with additional related dependencies, install the
Element with the `nwb` option as follows:

```console
pip install element-array-ephys[nwb]
```

## Roadmap

Incorporation of SpikeInterface into the Array Electrophysiology Element will be
on DataJoint Elements development roadmap. Dr. Loren Frank has led a development
effort of a DataJoint pipeline with SpikeInterface framework and
NeurodataWithoutBorders format integrated
[https://github.com/LorenFrankLab/nwb_datajoint](https://github.com/LorenFrankLab/nwb_datajoint).

Future additions to this element will add functionality to support large (> 48
hours) neuropixel recordings via an overlapping segmented processing approach. 

Further development of this Element is community driven. Upon user requests we will
continue adding features to this Element.
