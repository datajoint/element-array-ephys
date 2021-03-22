# Array Electrophysiology Element
## Description of modality, user population 

Neuropixels probes were developed by a collaboration between HHMI Janelia, industry partners, and others (Jun et al., Nature 2017). 
Since its initial release in October 2018, 300 labs have ordered 1200 probes. 
Since the rollout of Neuropixels 2.0 in October 2020, IMEC has been shipping 100+ probes monthly (correspondence with Tim Harris). 

Neuropixels probes offer 960 electrode sites along a 10mm long shank, 
with 384 recordable channels per probe that can record hundreds of units spanning multiple brain regions 
(Neuropixels 2.0 version is a 4-shank probe with 1280 electrode sites per shank). 
Such large recording capacity have offered tremendous opportunities for the field of neurophysiology research, 
yet this is accompanied by an equally great challenge in terms of data and computation management.

## Acquisition tools
The typical instrumentation used for data acquisition is the Neuropixels probe and headstage interfacing with a PXIe acquisition module (https://www.neuropixels.org/control-system).
Two main acquisition softwares are used for Neuropixels:
SpikeGLX - developed by Bill Karsh and Tim Harris at HHMI/Janelia
OpenEphys - developed by Joshua Siegle at the Allen Institute.
These save the data into specific directory structure and file-naming convention as custom binary formats (e.g. “.bin”, “.dat”). Meta data are stored as separate files in xml or text format.

## Preprocessing tools
The preprocessing pipeline includes bandpass filtering for LFP extraction, bandpass filtering for spike sorting,  spike sorting, and manual curation of the spike sorting results, and calculation of quality control metrics. In trial-based experiments, the spike trains are aligned and separated into trials. Standard processing may include PSTH computation aligned to trial onset or other events, and often grouped by different trial types. 
Neuroscience groups have traditionally developed custom home-made toolchains. 

In recent years, several leaders have been emerging as de facto standards with significant community uptake: 
+ Kilosort
+ JRClust
+ MountainSort
+ SpyKING CIRCUS

Kilosort provides most automation and has gained significant popularity, being adopted as one of the key spike sorting methods in the majority of the teams/collaborations we have worked with. As part of Year-1 U24 effort, we provide support for data ingestion of spike sorting results from Kilosort.
Further effort will be devoted for the ingestion support of other spike sorting methods. On this end, a framework for unifying existing spike sorting methods, named SpikeInterface, has been developed by Alessio Buccino, et al. SpikeInterface provides a convenient Python-based wrapper to invoke, extract, compare spike sorting results from different sorting algorithms.

## Precursor projects and interviews

Over the past few years, several labs have developed DataJoint-based data management and processing pipelines for Neuropixels probes. 
Our team collaborated with several of them during their projects. 
Additionally, we interviewed these teams to understand their experiment workflow, pipeline design, associated tools, and interfaces. 
These teams include:

+ International Brain Lab - https://internationalbrainlab.org - https://github.com/int-brain-lab/IBL-pipeline
+ Mesoscale Activity Project (HHMI Janelia) - https://github.com/mesoscale-activity-map - https://github.com/mesoscale-activity-map/map-ephys 
+ Moser Group (private repository)

## Pipeline Development
Through our interviews and direct collaboration on the precursor projects,
 we identified the common motifs to create the Neuropixels Element with the repository hosted at https://github.com/datajoint/elements-neuropixels.

Major features of the Neuropixels Element include:
+ Pipeline architecture defining:
    + Probe, electrode configuration compatible with Neuropixels probes and generalizable to other types of probes (e.g. tetrodes) - supporting both chronic and acute probe insertion mode
    + Probe-insertion, ephys-recordings, LFP extraction, clusterings, curations, sorted units and the associated data (e.g. spikes, waveforms, etc.)
    + Store/track/manage different curations of the spike sorting results
+ Ingestion support for data acquired with SpikeGLX and OpenEphys acquisition systems
+ Ingestion support for spike sorting outputs from Kilosort
+ Sample data and complete test suite for quality assurance

Incorporation of SpikeInterface into the Neuropixels Element will be on DataJoint Elements development roadmap. 
Dr. Loren Frank has led a development effort of a DataJoint pipeline with SpikeInterface framework and NeurodataWithoutBorders format integrated (https://github.com/LorenFrankLab/nwb_datajoint).

## Alpha release: Validation sites
We engaged several labs to adopt and validate the Neuropixels Element during the alpha testing phase. These include:
+ Andreas Tolias Lab (BCM) - Anthony Ramos and Saumil Patel
    + The lab uses Neuropixels 1.0 probes, recording in rodents and monkeys in the near future
    + The lab has an existing animal, lab management and imaging pipeline, in need of a Neuropixels ephys pipeline
    + The lab uses OpenEphys acquisition system, we extended the Neuropixels Element to provide support for OpenEphys in order to facilitate adoption/validation with Tolias Lab
    + The lab uses on-premise institutional resource for data infrastructure and hosting
    + Repository: https://github.com/cajal/pipeline 
+ BrainCoGs (Princeton Neuroscience Institute) - Manuel Schottdorf, Alvaro Luna
    + DataJoint NEURO conducted a training workshop on 3/01- 03. 
    + The Neuropixels Element was adopted, connected to an existing BrainCoGs project pipeline. The ingestion routine was successfully validated on the existing Neuropixels dataset (acquired with SpikeGLX) and Kilosort outputs.
    + Manuel Schottdorf expressed interest in comparison study between different spike sorting algorithms, thus further validating our roadmap to integrate the SpikeInterface framework into the Neuropixels Element
    + The lab uses local institutional resource for data infrastructure and hosting
    + Repository: https://github.com/BrainCOGS/U19-pipeline_python
+ Brody Lab (Princeton) - Adrian Bondy, Alvaro Luna
    + The Brody Lab performs ephys recordings on chronically inserted Neuropixels and tetrode probes. The lab is in the process of converting their existing MySQL database server into a DataJoint-compatible pipeline, and is in need of an ephys pipeline.
    + We have extended the Neuropixels Element and provided a design version capable of supporting chronic probe insertion experiments.
    + Though the ingestion routines are not included for non-Neuropixels probes, the Neuropixels Element’s architecture is compatible to ingest existing tetrode data acquired previously by the Brody Lab
    + The lab uses local institutional resource for data infrastructure and hosting
    + Repository: private repository

## Beta release
As the validation progresses, we expect to produce a beta version of the workflow for users to adopt independently by May 1, 2021.
