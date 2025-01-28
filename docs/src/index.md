# Element Array Electrophysiology

This Element features DataJoint schemas for analyzing extracellular array
electrophysiology data acquired with Neuropixels probes and spike sorted using [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).
Each Element is a modular pipeline for data storage and processing with
corresponding database tables that can be combined with other Elements to assemble a
fully functional pipeline.

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

The Element is comprised of `probe` and `ephys` schemas. Visit the 
[Concepts page](./concepts.md) for more information about the `probe` and
`ephys` schemas and an explanation of the tables. To get started with building your own
data pipeline, visit the [Tutorials page](./tutorials/index.md).

Prior to version `0.4.0` , several `ephys` schemas were
developed and supported to handle various use cases of this pipeline and workflow. These
 are now deprecated but still available on their own branch within the repository:

* [`ephys_acute`](https://github.com/datajoint/element-array-ephys/tree/main_ephys_acute)
* [`ephys_chronic`](https://github.com/datajoint/element-array-ephys/tree/main_ephys_chronic)
* [`ephys_precluster`](https://github.com/datajoint/element-array-ephys/tree/main_ephys_precluster)
* [`ephys_no_curation`](https://github.com/datajoint/element-array-ephys/tree/main_ephys_no_curation)
