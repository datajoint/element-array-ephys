from os import path

from setuptools import find_packages, setup

pkg_name = next(p for p in find_packages() if "." not in p)
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open(path.join(here, "requirements_dev.txt")) as f:
    requirements_dev = f.read().splitlines()

with open(path.join(here, "element_array_ephys/export/nwb/requirements.txt")) as f:
    requirements_nwb = f.read().splitlines()

with open(path.join(here, pkg_name, "version.py")) as f:
    exec(f.read())

setup(
    name=pkg_name.replace("_", "-"),
    version=__version__,  # noqa F821
    description="DataJoint Element for Extracellular Array Electrophysiology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DataJoint",
    author_email="info@datajoint.com",
    license="MIT",
    url=f'https://github.com/datajoint/{pkg_name.replace("_", "-")}',
    keywords="neuroscience electrophysiology science datajoint",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    extras_require={"dev": requirements_dev, "nwb": requirements_nwb},
    scripts=[],
    install_requires=requirements,
)
