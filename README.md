[![PyPI version](https://badge.fury.io/py/element-array-ephys.svg)](http://badge.fury.io/py/element-array-ephys)

# DataJoint Element for Extracellular Electrophysiology

DataJoint Element for extracellular array electrophysiology that processes data 
acquired with a polytrode probe
(e.g. [Neuropixels](https://www.neuropixels.org), Neuralynx) using the
[SpikeGLX](https://github.com/billkarsh/SpikeGLX) or
[OpenEphys](https://open-ephys.org/gui) acquisition software and 
[MATLAB-based Kilosort](https://github.com/MouseLand/Kilosort) or [python-based
Kilosort](https://github.com/MouseLand/pykilosort) spike sorting software. DataJoint 
Elements collectively standardize and automate data collection and analysis for 
neuroscience experiments. Each Element is a modular pipeline for data storage and 
processing with corresponding database tables that can be combined with other Elements 
to assemble a fully functional pipeline.

## Experiment flowchart

![flowchart](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

## Data Pipeline Diagram

![datajoint](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/attached_array_ephys_element_acute.svg)


## Getting Started

+ Install from PyPI

     ```bash
     pip install element-array-ephys
     ```

+ [Interactive tutorial on GitHub Codespaces](https://github.com/datajoint/workflow-array-ephys#interactive-tutorial)

+ [Documentation](https://datajoint.com/docs/elements/element-array-ephys)

## Support

+ If you need help getting started or run into any errors, please contact our team by email at support@datajoint.com.
