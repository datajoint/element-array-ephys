# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and
 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [0.2.11] - 2023-06-29

+ Update - Improve kilosort triggering routine - better logging, remove temporary files, robust resumable processing
+ Add - Null value for `package_version` to patch bug
+ Update - GitHub Actions workflows
+ Update - README instructions

## [0.2.10] - 2023-05-26

+ Add - Kilosort, NWB, and DANDI citations
+ Fix - CSS to improve readability of tables in dark mode
+ Update - mkdocs.yaml

## [0.2.9] - 2023-05-11

+ Fix - `.ipynb` dark mode output for all notebooks.

## [0.2.8] - 2023-04-28

+ Fix - `.ipynb` output in tutorials is not visible in dark mode.

## [0.2.7] - 2023-04-19

+ Bugfix - A name remapping dictionary was added to ensure consistency between the column names of the `metrics.csv` file and the attribute names of the `QualityMetrics` table

## [0.2.6] - 2023-04-17

+ Fix - Update Pandas DataFrame column name to insert `pt_ratio` in `QualityMetrics.Waveform` table

## [0.2.5] - 2023-04-12

+ Add - docstrings for quality metric tables
+ Fix - docstring errors
+ Update - `concepts.md`
+ Update - schema diagrams with quality metrics tables

## [0.2.4] - 2023-03-10

+ Update - Requirements with `ipywidgets` and `scikit-image` for plotting widget

## [0.2.3] - 2023-02-14

+ Add - extras_require install options for nwb and development requirement sets
+ Add - mkdocs notebook rendering
+ Add - markdown linting and spellcheck config files, with implementation edits
+ Update - license for 2023
+ Update - blackify previous updates

## [0.2.2] - 2022-01-11

+ Bugfix - Revert import order in `__init__.py` to avoid circular import error.
+ Update - `.pre-commit-config.yaml` to disable automatic positioning of import
  statement at the top.
+ Bugfix - Update docstrings to render API for documentation website.

## [0.2.1] - 2022-01-06

+ Add - `build_electrode_layouts` function in `probe.py` to compute the electrode layout
  for all types of probes.
+ Update - parameterize run_CatGT step from parameters retrieved from
  `ClusteringParamSet` table
+ Update - clustering step, update duration for "median_subtraction" step
+ Bugfix - handles single probe recording in "Neuropix-PXI" format
+ Update - safeguard in creating/inserting probe types upon probe activation
+ Add - quality control metric dashboard
+ Update & fix docstrings
+ Update - `ephys_report.UnitLevelReport` to add `ephys.ClusterQualityLabel` as a
  foreign key reference
+ Add - `.pre-commit-config.yaml`

## [0.2.0] - 2022-10-28

+ Add - New schema `ephys_report` to compute and store figures from results
+ Add - Widget to display figures
+ Add - Add `ephys_no_curation` and routines to trigger spike-sorting analysis
  using Kilosort (2.0, 2.5)
+ Add - mkdocs for Element Documentation
+ Add - New `QualityMetrics` table to store clusters' and waveforms' metrics after the
  spike sorting analysis.

## [0.1.4] - 2022-07-11

+ Bugfix - Handle case where `spike_depths` data is present.

## [0.1.3] - 2022-06-16

+ Update - Allow for the `precluster_output_dir` attribute to be nullable when no
  pre-clustering is performed.

## [0.1.2] - 2022-06-09

+ Bugfix - Handle case where `pc_features.npy` does not exist.

## [0.1.1] - 2022-06-01

+ Add - Secondary attributes to `PreClusterParamSteps` table

## [0.1.0] - 2022-05-26

+ Update - Rename module for acute probe insertions from `ephys.py` to `ephys_acute.py`.
+ Add - Module for pre-clustering steps (`ephys_precluster.py`), which is built off of
  `ephys_acute.py`.
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

[0.2.10]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.10
[0.2.9]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.9
[0.2.8]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.8
[0.2.7]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.7
[0.2.6]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.6
[0.2.5]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.5
[0.2.4]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.4
[0.2.3]: https://github.com/datajoint/element-array-ephys/releases/tag/0.2.3
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
