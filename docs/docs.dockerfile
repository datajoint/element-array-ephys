FROM python:latest

WORKDIR /main
COPY . /main
RUN pip install mkdocs-material mkdocs-redirects mkdocstrings mkdocstrings-python