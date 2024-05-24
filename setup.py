from os import path

from setuptools import find_packages, setup

pkg_name = "element_array_ephys"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

with open(path.join(here, "element_array_ephys/export/nwb/requirements.txt")) as f:
    requirements_nwb = f.read().splitlines()

with open(path.join(here, pkg_name, "version.py")) as f:
    exec(f.read())

setup(
    name=pkg_name.replace("_", "-"),
    python_requires=">=3.7, <3.11",
    version=__version__,  # noqa F821
    description="Extracellular Array Electrophysiology DataJoint Element",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DataJoint",
    author_email="info@datajoint.com",
    license="MIT",
    url=f'https://github.com/datajoint/{pkg_name.replace("_", "-")}',
    keywords="neuroscience electrophysiology science datajoint",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    scripts=[],
    install_requires=[
        "datajoint>=0.13.0",
        "ipykernel>=6.0.1",
        "ipywidgets",
        "openpyxl",
        "plotly",
        "seaborn",
        "spikeinterface @ git+https://github.com/SpikeInterface/spikeinterface.git",
        "scikit-image",
        "nbformat>=4.2.0",
        "pyopenephys>=1.1.6",
        "element-interface @ git+https://github.com/datajoint/element-interface.git",
        "numba",
    ],
    extras_require={
        "elements": [
            "element-animal @ git+https://github.com/datajoint/element-animal.git",
            "element-event @ git+https://github.com/datajoint/element-event.git",
            "element-lab @ git+https://github.com/datajoint/element-lab.git",
            "element-session @ git+https://github.com/datajoint/element-session.git",
        ],
        "nwb": ["dandi", "neuroconv[ecephys]", "pynwb"],
        "tests": ["pre-commit", "pytest", "pytest-cov"],
        "spikingcircus": ["hdbscan"],
    },
)
