# DataJoint Element - Array Electrophysiology Element

This repository features DataJoint pipeline design for extracellular array electrophysiology, 
with ***Neuropixels*** probe and ***kilosort*** spike sorting method. 

The pipeline presented here is not a complete pipeline by itself, but rather a modular 
design of tables and dependencies specific to the extracellular electrophysiology workflow. 

This modular pipeline element can be flexibly attached downstream 
to any particular design of experiment session, thus assembling a fully functional 
ephys pipeline.

See [Background](Background.md) for the background information and development timeline.

## Element architecture

![element-array-ephys diagram](images/attached_array_ephys_element.svg)

As the diagram depicts, the array ephys element starts immediately downstream from ***Session***, 
and also requires some notion of ***Location*** as a dependency for ***InsertionLocation***. We 
provide an [example workflow](https://github.com/datajoint/workflow-array-ephys/) with a 
[pipeline script](https://github.com/datajoint/workflow-array-ephys/blob/main/workflow_array_ephys/pipeline.py)
that models (a) combining this Element with the corresponding [Element-Session](https://github.com/datajoint/element-session)
, and (b) declaring a ***SkullReference*** table to provide Location.

### The design of probe

+ ***ProbeType*** - a lookup table specifying the type of Neuropixels probe (e.g. "neuropixels 1.0", "neuropixels 2.0 single-shank")
+ ***ProbeType.Electrode*** - all electrode and their properties for a particular probe type
    + An electrode here refers to one recordable electrode site on the Neuropixels probe (e.g. for Neuropixels 1.0, there are 960 sites per shank)
+ ***Probe*** - record of an actual physical probe, identifiable by some unique ID (e.g. probe's serial number)
+ ***ElectrodeConfig*** - particular electrode configuration to be used for ephys recording
+ ***ElectrodeConfig.Electrode*** - corresponding electrodes in ***ProbeType.Electrode*** that are used for recording in this electrode configuration (e.g. for Neuropixels 1.0 or 2.0, there can be at most 384 electrodes usable for recording per probe)

### Extracellular ephys recording

+ ***ProbeInsertion*** - a surgical insertion of a probe in the brain. Every experimental session consists of one or more entries in ***ProbeInsertion*** with a corresponding ***InsertionLocation*** each
+ ***EphysRecording*** - each ***ProbeInsertion*** is accompanied by a corresponding ***EphysRecording***, specifying the ***ElectrodeConfig*** used for the recording from the ***Probe*** defined in such ***ProbeInsertion***
    
### Clusters and spikes

This ephys element features automatic ingestion for spike sorting results from the ***kilosort*** method. 

+ ***Clustering*** - specify instance(s) of clustering on an ***EphysRecording***, by some ***ClusteringMethod***
+ ***Curation*** - specify instance(s) of curations performed on the output of a given ***Clustering***
+ ***CuratedClustering*** - set of results from a particular round of clustering/curation
    + ***CuratedClustering.Unit*** - Identified unit(s) from one ***Curation***, and the associated properties (e.g. cluster quality, spike times, spike depths, etc.)
    + ***WaveformSet*** - A set of spike waveforms for units from a given CuratedClustering

## Installation

+ Install `element-array-ephys`
    ```
    pip install element-array-ephys
    ```

+ Upgrade `element-array-ephys` previously installed with `pip`
    ```
    pip install --upgrade element-array-ephys
    ```

+ Install `element-interface`

    + `element-interface` is a dependency of `element-array-ephys`, however it is not contained within `requirements.txt`.
     
    ```
    pip install "element-interface @ git+https://github.com/datajoint/element-interface"
    ```

## Usage

### Element activation

To activate the `element-array-ephys`, ones need to provide:

1. Schema names
    + schema name for the probe module
    + schema name for the ephys module

2. Upstream tables
    + Session table: A set of keys identifying a recording session (see [Element-Session](https://github.com/datajoint/element-session)).
    + SkullReference table: A reference table for InsertionLocation, specifying the skull reference (see [example pipeline](https://github.com/datajoint/workflow-array-ephys/blob/main/workflow_array_ephys/pipeline.py)).

3. Utility functions. See [example definitions here](https://github.com/datajoint/workflow-array-ephys/blob/main/workflow_array_ephys/paths.py)
    + get_ephys_root_data_dir(): Returns your root data directory.
    + get_session_directory(): Returns the path of the session data relative to the root.

For more detail, check the docstring of the `element-array-ephys`:

    help(probe.activate)
    help(ephys.activate)

### Example usage

See [this project](https://github.com/datajoint/workflow-array-ephys) for an example usage of this Array Electrophysiology Element.
