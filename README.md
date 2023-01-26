# DataJoint Element - Array Electrophysiology Element

DataJoint Element for extracellular array electrophysiology. DataJoint Elements
collectively standardize and automate data collection and analysis for neuroscience
experiments.  Each Element is a modular pipeline for data storage and processing with
corresponding database tables that can be combined with other Elements to assemble a
fully functional pipeline.

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

Installation and usage instructions can be found at the
[Element documentation](https://datajoint.com/docs/elements/element-array-ephys).


## The "modular clustering" branch

Note: this `modular clustering` branch is experimental, to be used at users' own risk. Contact [DataJoint team](https://datajoint.com/) for questions or further instruction.

This `modular clustering` branch features a separate set of DataJoint pipeline dedicated for running spike-sorting using the [ecephys_spike_sorting](https://github.com/datajoint-company/ecephys_spike_sorting) flow.
(see details [here](./element_array_ephys/spike_sorting/ecephys_spike_sorting.py))

The new `ecephys` pipeline is designed to work in conjunction with the default ephys pipeline, replacing the execution of the `make()` of the `ephys.Clustering` table.
Thus, upon activation and incorporation of this `ecephys` pipeline to a workflow, users need to modify the `key_source` of the `ephys.Clustering` table to enable this behavior.

```python
    from element_array_ephys import probe
    from element_array_ephys import ephys_no_curation as ephys
    from element_array_ephys.spike_sorting import ecephys_spike_sorting as ephys_sorter
    
    from workflow.pipeline import reference, session  # some existing "session" and "reference" schemas 
    
    Session = session.Session
    SkullReference = reference.SkullReference
    
    ephys.activate("ephys", "probe", linking_module=__name__)
    ephys_sorter.activate("ephys_sorter", ephys_module=ephys)
    
    ephys.Clustering.key_source = (
        ephys.Clustering.key_source - ephys_sorter.KilosortPreProcessing.key_source
    ).proj()
```

