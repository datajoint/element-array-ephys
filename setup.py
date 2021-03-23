from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = """"
DataJoint Element for Extracellular Array Electrophysiology with Neuropixels probe
"""

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='element-array-ephys',
    version='0.0.1',
    description="DataJoint Element for Extracellular Array Electrophysiology",
    long_description=long_description,
    author='DataJoint NEURO',
    author_email='info@vathes.com',
    license='MIT',
    url='https://github.com/datajoint/element-array-ephys',
    keywords='neuroscience electrophysiology science datajoint',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    scripts=[],
    install_requires=requirements,
)
