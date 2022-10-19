# Element Array Electrophysiology

This repository features DataJoint schemas for analyzing extracellular array electrophysiology acquired with Neuropixels probes and spike sorted using Kilosort spike sorter. 

The Element is comprised of `probe` and `ephys` schemas. There are several `ephys` schemas including `ephys_acute`, `ephys_chronic`, and `ephys_precluster` to handle several use cases of this pipeline and workflow. For more information about the use cases of ephys schemas and an explanation of the tables, see the [concepts page](./concepts.md) 

### `ephys_acute` module
![element-array-ephys-acute diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_acute.svg)

### `ephys_chronic` module
![element-array-ephys-chronic diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_chronic.svg)

### `ephys_precluster` module
![element-array-ephys-precluster diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_precluster.svg)

## Citation

If your work DataJoint Elements, please cite the following manuscript and Research Resource Identifier (RRID).

+ Yatsenko D, Nguyen T, Shen S, Gunalan K, Turner CA, Guzman R, Sasaki M, Sitonic D, Reimer J, Walker EY, Tolias AS. DataJoint Elements: Data Workflows for Neurophysiology. bioRxiv. 2021 Jan 1. doi: https://doi.org/10.1101/2021.03.30.437358

+ DataJoint Elements ([RRID:SCR_021894](https://scicrunch.org/resolver/SCR_021894)) - Element Array Electrophysiology (version `<Enter version number>`)