# Tutorials

## Installation

Installation of the Element requires an integrated development environment and database.
Instructions to setup each of the components can be found on the
[User Instructions](https://datajoint.com/docs/elements/user-guide/) page.  These
instructions use the example
[workflow for Element Array Ephys](https://github.com/datajoint/workflow-array-ephys),
which can be modified for a user's specific experimental requirements.  This example
workflow uses several Elements (Lab, Animal, Session, Event, and Electrophysiology) to construct
a complete pipeline, and is able to ingest experimental metadata and run model training
and inference.

### Videos

The [Element Array Ephys tutorial](https://youtu.be/KQlGYOBq7ow?t=3658) gives an
overview of the workflow files and notebooks as well as core concepts related to
Electrophysiology.

[![YouTube tutorial](https://img.youtube.com/vi/KQlGYOBq7ow/0.jpg)](https://youtu.be/KQlGYOBq7ow?t=3658)

### Notebooks

Each of the notebooks in the workflow
([download here](https://github.com/datajoint/workflow-array-ephys/tree/main/notebooks)
steps through ways to interact with the Element itself. For convenience, these notebooks
are also rendered as part of this site. To try out the Elements notebooks in an online
Jupyter environment with access to example data, visit
[CodeBook](https://codebook.datajoint.io/). (Electrophysiology notebooks coming soon!)

- [Data Download](./00-data-download-optional.ipynb) highlights how to use DataJoint
  tools to download a sample model for trying out the Element.
- [Configure](./01-configure.ipynb) helps configure your local DataJoint installation to
  point to the correct database.
- [Workflow Structure](./02-workflow-structure-optional.ipynb) demonstrates the table
   architecture of the Element and key DataJoint basics for interacting with these
   tables.
- [Process](./03-process.ipynb) steps through adding data to these tables and launching
   key Electrophysiology features, like model training.
- [Automate](./04-automate-optional.ipynb) highlights the same steps as above, but
  utilizing all built-in automation tools.
- [Explore](./05-explore.ipynb) demonstrates how to fetch data from the Element.
- [Drop schemas](./06-drop-optional.ipynb) provides the steps for dropping all the
  tables to start fresh.
- [Downstream Analysis](./07-downstream-analysis.ipynb) highlights how to link
   this Element to Element Event for event-based analyses.
- [Visualizations](./10-data_visualization.ipynb) highlights how to use a built-in module
   for visualizing units, probes and quality metrics.
- [Electrode Localization](./08-electrode-localization.ipynb) demonstrates how to link
  this Element to
  [Element Electrode Localization](https://datajoint.com/docs/elements/element-electrode-localization/).
- [NWB Export](./09-NWB-export.ipynb) highlights the export functionality available for the
  `no-curation` schema.
