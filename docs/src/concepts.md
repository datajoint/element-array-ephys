# Concepts

## Acquisition Tools for Electrophysiology

### Neuropixels Probes
Neuropixels probes were developed by a collaboration between HHMI Janelia, industry partners, and others (Jun et al., Nature 2017). Since their initial release in October 2018, 300 labs have ordered 1200 probes. Since the rollout of Neuropixels 2.0 in October 2020, IMEC has been shipping 100+ probes monthly (correspondence with Tim Harris).

Neuropixels probes offer 960 electrode sites along a 10mm long shank, with 384 recordable channels per probe that can record hundreds of units spanning multiple brain regions (Neuropixels 2.0 version is a 4-shank probe with 1280 electrode sites per shank). Such large recording capacity has offered tremendous opportunities for the field of neurophysiology research, yet this is accompanied by an equally great challenge in terms of data and computation management.

### Data Acquisition Tools
The typical instrumentation used for data acquisition is the Neuropixels probe and headstage interfacing with a [PXIe acquisition module](https://www.neuropixels.org/control-system). Two main acquisition softwares are used for Neuropixels:

+ SpikeGLX - developed by Bill Karsh and Tim Harris at HHMI/Janelia
+ OpenEphys - developed by Joshua Siegle at the Allen Institute.

These save the data into a specific directory structure and file-naming convention with custom binary formats (e.g. “.bin”, “.dat”). Meta data are stored as separate files in xml or text format.

### Data Preprocessing Tools
The preprocessing pipeline includes bandpass filtering for LFP extraction, bandpass filtering for spike sorting, spike sorting, manual curation of the spike sorting results, and calculation of quality control metrics. In trial-based experiments, the spike trains are aligned and separated into trials. Standard processing may include PSTH computation aligned to trial onset or other events, and often grouped by different trial types. Neuroscience groups have traditionally developed custom home-made toolchains.

In recent years, several leaders have been emerging as de facto standards with significant community uptake:

+ Kilosort
+ JRClust
+ MountainSort
+ SpyKING CIRCUS

Kilosort provides most automation and has gained significant popularity, being adopted as one of the key spike sorting methods in the majority of the teams/collaborations we have worked with. As part of our Year-1 NIH U24 effort, we provide support for data ingestion of spike sorting results from Kilosort. Further effort will be devoted for the ingestion support of other spike sorting methods. On this end, a framework for unifying existing spike sorting methods, named [SpikeInterface](https://github.com/SpikeInterface/spikeinterface), has been developed by Alessio Buccino, et al. SpikeInterface provides a convenient Python-based wrapper to invoke, extract, compare spike sorting results from different sorting algorithms.

## Key Partnerships

Over the past few years, several labs have developed DataJoint-based data management and processing pipelines for Neuropixels probes. Our team collaborated with several of them during their projects. Additionally, we interviewed these teams to understand their experimental workflow, pipeline design, associated tools, and interfaces. These teams include:

- [International Brain Lab](https://internationalbrainlab.org) - [https://github.com/int-brain-lab/IBL-pipeline](https://github.com/int-brain-lab/IBL-pipeline)

- [Mesoscale Activity Project (HHMI Janelia)](https://github.com/mesoscale-activity-map) - [https://github.com/mesoscale-activity-map/map-ephys](https://github.com/mesoscale-activity-map/map-ephys)

- Moser Group (Norwegian University of Science and Technology) - see [pipeline design](https://moser-pipelines.readthedocs.io/en/latest/ephys/overview.html)

- Andreas Tolias Lab (Baylor College of Medicine)

- BrainCoGs (Princeton Neuroscience Institute)

- Brody Lab (Princeton University)

## Element Architecture

Each of the DataJoint Elements are a set of tables for common neuroinformatics modalities to organize, preprocess, and analyze data. Each node in the following diagram is either a table in the Element itself or a table that would be connected to the Element.

![element-array-ephys diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_acute.svg)

### `subject` schema ([API docs](https://datajoint.com/docs/elements/element-animal/api/element_animal/subject))
- Although not required, most choose to connect the `Session` table to a `Subject` table.

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject |

### `session` schema ([API docs](https://datajoint.com/docs/elements/element-session/api/element_session/session_with_datetime))

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier |

### `probe` schema ([API docs](../api/element_array_ephys/probe))
Tables related to the Neuropixels probe and electrode configuration.

| Table | Description |
| --- | --- |
| ProbeType | a lookup table specifying the type of Neuropixels probe (e.g. "neuropixels 1.0", "neuropixels 2.0 single-shank"). |
| ProbeType.Electrode |  all electrodes and their properties for a particular probe type. |
| Probe | record of an actual physical probe, identifiable by some unique ID (e.g. probe's serial number). |
| ElectrodeConfig | particular electrode configuration to be used for ephys recording. |
| ElectrodeConfig.Electrode | corresponding electrodes in `ProbeType.Electrode` that are used for recording in this electrode configuration (e.g. for Neuropixels 1.0 or 2.0, there can be at most 384 electrodes usable for recording per probe). |

### `ephys` schema ([API docs](../api/element_array_ephys/ephys))
Tables related to information about physiological recordings and automatic ingestion of spike sorting results.

| Table | Description |
| --- | --- |
| ProbeInsertion | A surgical insertion of a probe in the brain. Every experimental session consists of one or more entries in `ProbeInsertion` with a corresponding `InsertionLocation` each. |
| EphysRecording | each `ProbeInsertion` is accompanied by a corresponding `EphysRecording`, specifying the `ElectrodeConfig` used for the recording from the `Probe` defined in `ProbeInsertion`. |
| Clustering | specify instance(s) of clustering on an `EphysRecording`, by some `ClusteringMethod`. |
| Curation | specify instance(s) of curations performed on the output of a given `Clustering`. |
| CuratedClustering | set of results from a particular round of clustering/curation. |
| CuratedClusting.Unit | Identified unit(s) from one `Curation`, and the associated properties (e.g. cluster quality, spike times, spike depths, etc.). |
| WaveformSet | A set of spike waveforms for units from a given `CuratedClustering`. |

## Element Development

Through our interviews and direct collaboration on the precursor projects, we identified the common motifs to create the [Array Electrophysiology Element](https://github.com/datajoint/element-array-ephys).

Major features of the Array Electrophysiology Element include:

+ Pipeline architecture detailed by:

    + Probe, electrode configuration compatible with Neuropixels probes and generalizable to other types of probes (e.g. tetrodes) - supporting both `chronic` and `acute` probe insertion modes.

    + Probe-insertion, ephys-recordings, LFP extraction, clusterings, curations, sorted units and the associated data (e.g. spikes, waveforms, etc.).
    
    + Store/track/manage different curations of the spike sorting results - supporting both curated clustering and kilosort triggered clustering (i.e., `no_curation`).

+ Ingestion support for data acquired with SpikeGLX and OpenEphys acquisition systems. 
+ Ingestion support for spike sorting outputs from Kilosort.
+ Triggering support for workflow integrated Kilosort processing.
+ Sample data and complete test suite for quality assurance.

Incorporation of SpikeInterface into the Array Electrophysiology Element will be on DataJoint Elements development roadmap. Dr. Loren Frank has led a development effort of a DataJoint pipeline with SpikeInterface framework and NeurodataWithoutBorders format integrated (https://github.com/LorenFrankLab/nwb_datajoint).

## Data Export and Publishing

Element Array Electrophysiology supports exporting of all data into standard Neurodata Without Borders (NWB) files. This makes it easy to share files with collaborators and publish results on [DANDI Archive](https://dandiarchive.org/). [NWB](https://www.nwb.org/), as an organization, is dedicated to standardizing data formats and maximizing interoperability across tools for neurophysiology. For more information on uploading NWB files to DANDI within the DataJoint Elements ecosystem, visit our documentation for the DANDI upload feature of [Element Array Electrophysiology](datajoint.com/docs/elements/element-array-ephys/).

## Roadmap

Further development of this Element is community driven. Upon user requests we will continue adding features to this Element.