FROM python:slim

WORKDIR /main
COPY ./docs /main
COPY ./element_array_ephys /main
COPY requirements.txt /main
COPY setup.py /main
RUN pip install mkdocs-material mkdocs-redirects mkdocstrings mkdocstrings-python