from setuptools import setup, find_packages
from os import path

pkg_name = 'element_array_ephys'
here = path.abspath(path.dirname(__file__))

long_description = """"
DataJoint Element for Extracellular Array Electrophysiology with Neuropixels probe
"""

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join(here, pkg_name, 'version.py')) as f:
    exec(f.read())

setup(
    name='element-array-ephys',
    version=__version__,
    description="DataJoint Element for Extracellular Array Electrophysiology",
    long_description=long_description,
    author='DataJoint NEURO',
    author_email='info@vathes.com',
    license='MIT',
    url='https://github.com/ttngu207/element-array-ephys',
    keywords='neuroscience electrophysiology science datajoint',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    scripts=[],
    install_requires=requirements,
)
