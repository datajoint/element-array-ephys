from setuptools import setup, find_packages
from os import path

pkg_name = 'element-array-ephys'
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join(here, pkg_name.replace('-', '_'), 'version.py')) as f:
    exec(f.read())

setup(
    name=pkg_name,
    version=__version__,
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
