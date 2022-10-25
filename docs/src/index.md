# Element Array Electrophysiology

This Element features DataJoint schemas for analyzing extracellular array electrophysiology data acquired with Neuropixels probes and spike sorted using Kilosort spike sorter. Each Element is a modular pipeline for data storage and processing with corresponding database tables that can be combined with other Elements to assemble a fully functional pipeline.

The Element is comprised of `probe` and `ephys` schemas. There are several `ephys` schemas including `ephys_acute`, `ephys_chronic`, and `ephys_precluster` to handle several use cases of this pipeline and workflow. Visit the [Concepts page](./concepts.md) for more information about the use cases of `ephys` schemas and an explanation of the tables. To get started with building your own data pipeline, visit the [Tutorials page](./tutorials.md).  

### `ephys_acute` module
![element-array-ephys-acute diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_acute.svg)

### `ephys_chronic` module
![element-array-ephys-chronic diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_chronic.svg)

### `ephys_precluster` module
![element-array-ephys-precluster diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_precluster.svg)