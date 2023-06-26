[![PyPI version](https://badge.fury.io/py/element-array-ephys.svg)](http://badge.fury.io/py/element-array-ephys)

# DataJoint Element - Array Electrophysiology

DataJoint Element for extracellular array electrophysiology. DataJoint Elements
collectively standardize and automate data collection and analysis for neuroscience
experiments.  Each Element is a modular pipeline for data storage and processing with
corresponding database tables that can be combined with other Elements to assemble a
fully functional pipeline.

![diagram](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

Installation and usage instructions can be found at the
[Element documentation](https://datajoint.com/docs/elements/element-array-ephys).

DataJoint Workflow for extracellular array electrophysiology combines multiple DataJoint Elements to process data acquired with a polytrode probe
(e.g. [Neuropixels](https://www.neuropixels.org), Neuralynx) using the
[SpikeGLX](https://github.com/billkarsh/SpikeGLX) or
[OpenEphys](https://open-ephys.org/gui) acquisition software and processed with
[MATLAB-based Kilosort](https://github.com/MouseLand/Kilosort) or [python-based
Kilosort](https://github.com/MouseLand/pykilosort) spike sorting software. DataJoint Elements collectively standardize and automate data collection and analysis for neuroscience experiments. Each Element is a modular pipeline for data storage and processing with corresponding database tables that can be combined with other Elements to assemble a fully functional pipeline.

To get started, see below for an [interactive tutorial](#interactive-tutorial) on GitHub Codespaces.  More information can be found at the
[Element documentation page](https://datajoint.com/docs/elements/element-array-ephys).

## Experiment Flowchart

![flowchart](https://raw.githubusercontent.com/datajoint/element-array-ephys/main/images/diagram_flowchart.svg)

## Data Pipeline for Acute Experiment

![datajoint](https://raw.githubusercontent.com/datajoint/workflow-array-ephys/main/images/attached_array_ephys_element.svg)

## Interactive Tutorial on GitHub Codespaces

The easiest way to learn about DataJoint Elements is to use the tutorial notebooks within the included interactive environment configured using [DevContainer](https://containers.dev/).

### Launch Environment

- Launch using [GitHub Codespaces](https://github.com/features/codespaces) using the `+` option which will `Create codespace on main` in the codebase repository on your fork with default options. For more control, see the `...` where you may create `New with options...`.

- Cloud-based Development Environment: (*recommended*)
  - Launch using [GitHub Codespaces](https://github.com/features/codespaces) using the `+` option which will `Create codespace on main` in the codebase repository on your fork with default options. For more control, see the `...` where you may create `New with options...`.
  - Build time for a codespace is **~7m**. This is done infrequently and cached for convenience.
  - Start time for a codespace is **~30s**. This will pull the built codespace from cache when you need it.
  - *Tip*: Each month, GitHub renews a [free-tier](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#monthly-included-storage-and-core-hours-for-personal-accounts) quota of compute and storage. Typically we run into the storage limits before anything else since Codespaces consume storage while stopped. It is best to delete Codespaces when not actively in use and recreate when needed. We'll soon be creating prebuilds to avoid larger build times. Once any portion of your quota is reached, you will need to wait for it to be reset at the end of your cycle or add billing info to your GitHub account to handle overages.
  - *Tip*: GitHub auto names the codespace but you can rename the codespace so that it is easier to identify later.
- Local Development Environment:
  - Note: On Windows, running the tutorial notebook with the example data in a Dev Container is not currently possible due to a s3fs mounting issue.  Please use the `Cloud-based Development Environment` option above.
  - Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  - Install [Docker](https://docs.docker.com/get-docker/)
  - Install [VSCode](https://code.visualstudio.com/)
  - Install the VSCode [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  - `git clone` the codebase repository and open it in VSCode
  - Use the `Dev Containers extension` to `Reopen in Container` (More info in the `Getting started` included with the extension)

- Start time for a codespace is **~30s**. This will pull the built codespace from cache when you need it.

- You will know your environment has finished loading once you either see a terminal open related to `Running postStartCommand` with a final message: `Done` or the `README.md` is opened in `Preview`.

- *Tip*: Each month, GitHub renews a [free-tier](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#monthly-included-storage-and-core-hours-for-personal-accounts) quota of compute and storage. Typically we run into the storage limits before anything else since Codespaces consume storage while stopped. It is best to delete Codespaces when not actively in use and recreate when needed. We'll soon be creating prebuilds to avoid larger build times. Once any portion of your quota is reached, you will need to wait for it to be reset at the end of your cycle or add billing info to your GitHub account to handle overages.

- *Tip*: GitHub auto names the codespace but you can rename the codespace so that it is easier to identify later.

### Instructions

1. We recommend you start by navigating to the `notebooks` directory on the left panel and go through the `demo_prepare.ipynb` and `demo_run.ipynb` Jupyter notebooks. Execute the cells in the notebooks to begin your walk through of the tutorial.

1. Once you are done, see the options available to you in the menu in the bottom-left corner. For example, in Codespace you will have an option to `Stop Current Codespace` but when running DevContainer on your own machine the equivalent option is `Reopen folder locally`. By default, GitHub will also automatically stop the Codespace after 30 minutes of inactivity.

+ If you are new to GitHub and run into any errors, please contact us via email at support@datajoint.com. If you are experienced with GitHub, please create an issue on the upstream repository or issue a pull request with a thorough explanation of the error and proposed solution.
