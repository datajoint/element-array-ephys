# Element Array Electrophysiology

This Element features DataJoint schemas for analyzing extracellular array
electrophysiology data acquired with Neuropixels probes and spike sorted using Kilosort
spike sorter. Each Element is a modular pipeline for data storage and processing with
corresponding database tables that can be combined with other Elements to assemble a
fully functional pipeline.

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

The Element is comprised of `probe` and `ephys` schemas. Several `ephys` schemas are
developed to handle various use cases of this pipeline and workflow:

+ `ephys_acute`: A probe is inserted into a new location during each session.

+ `ephys_chronic`: A probe is inserted once and used to record across multiple
  sessions.

+ `ephys_precluster`: A probe is inserted into a new location during each session.
  Pre-clustering steps are performed on the data from each probe prior to Kilosort
  analysis.

+ `ephys_no_curation`: A probe is inserted into a new location during each session and
  Kilosort-triggered clustering is performed without the option to manually curate the
  results.

Visit the [Concepts page](./concepts.md) for more information about the use cases of
`ephys` schemas and an explanation of the tables. To get started with building your own
data pipeline, visit the [Tutorials page](./tutorials/index.md).
