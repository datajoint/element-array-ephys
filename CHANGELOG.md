# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and
 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [0.2.2] - 2022-01-11

+ Bugfix - Revert import order in `__init__.py` to avoid circular import error.
+ Update - `.pre-commit-config.yaml` to disable automatic positioning of import statement at the top.
+ Bugfix - Update docstrings to render API for documentation website.

## [0.2.1] - 2022-01-06

+ Add - `build_electrode_layouts` function in `probe.py` to compute the electrode layout for all types of probes.
+ Update - parameterize run_CatGT step from parameters retrieved from `ClusteringParamSet` table
+ Update - clustering step, update duration for "median_subtraction" step
+ Bugfix - handles single probe recording in "Neuropix-PXI" format
+ Update - safeguard in creating/inserting probe types upon probe activation
+ Add - quality control metric dashboard
+ Update & fix docstrings
+ Update - `ephys_report.UnitLevelReport` to add `ephys.ClusterQualityLabel` as a foreign key reference
+ Add - `.pre-commit-config.yaml`

## [0.2.0] - 2022-10-28

+ Add - New schema `ephys_report` to compute and store figures from results
+ Add - Widget to display figures
+ Add - Add `ephys_no_curation` and routines to trigger spike-sorting analysis
  using Kilosort (2.0, 2.5)
+ Add - mkdocs for Element Documentation
+ Add - New `QualityMetrics` table to store clusters' and waveforms' metrics after the spike sorting analysis.

## [0.1.4] - 2022-07-11

+ Bugfix - Handle case where `spike_depths` data is present.

## [0.1.3] - 2022-06-16

+ Update - Allow for the `precluster_output_dir` attribute to be nullable when no pre-clustering is performed.

## [0.1.2] - 2022-06-09

+ Bugfix - Handle case where `pc_features.npy` does not exist.

## [0.1.1] - 2022-06-01

+ Add - Secondary attributes to `PreClusterParamSteps` table

## [0.1.0] - 2022-05-26

+ Update - Rename module for acute probe insertions from `ephys.py` to `ephys_acute.py`.
+ Add - Module for pre-clustering steps (`ephys_precluster.py`), which is built off of `ephys_acute.py`.
+ Add - Module for chronic probe insertions (`ephys_chronic.py`).
+ Bugfix - Missing `fileTimeSecs` key in SpikeGLX meta file.
+ Update - Move common functions to `element-interface` package.
+ Add - NWB export function

## [0.1.0b4] - 2021-11-29

+ Add - Processing with Kilosort and pyKilosort for Open Ephys and SpikeGLX


## [0.1.0b0] - 2021-05-07

+ Update - First beta release

## [0.1.0a5] - 2021-05-05

+ Add - GitHub Action release process
+ Add - `probe` and `ephys` elements
+ Add - Readers for: `SpikeGLX`, `Open Ephys`, `Kilosort`
+ Add - Probe table supporting: Neuropixels probes 1.0 - 3A, 1.0 - 3B, 2.0 - SS,
  2.0 - MS


[0.2.2]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.2
[0.2.1]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.1
[0.2.0]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.0
[0.1.4]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.4
[0.1.3]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.3
[0.1.2]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.2
[0.1.1]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.1
[0.1.0]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.0
[0.1.0b4]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.0b4
[0.1.0b0]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.0b0
[0.1.0a5]: https://github.com/datajoint/element-array-ephys/releases/tag/0.1.0a5
