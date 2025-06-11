ARG PY_VER=3.11
ARG DISTRO=bullseye
FROM mcr.microsoft.com/devcontainers/python:${PY_VER}-${DISTRO}

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN \
    # install docker
    apt-get update && \
    apt-get install ca-certificates curl gnupg lsb-release -y && \
    mkdir -m 0755 -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y && \
    usermod -aG docker vscode && \
    apt-get clean

RUN \
    # dev setup
    apt update && \
    apt-get install sudo git bash-completion graphviz default-mysql-client s3fs procps -y && \
    usermod -aG sudo vscode && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    pip install --no-cache-dir --upgrade black pip nbconvert && \
    echo '. /etc/bash_completion' >> /home/vscode/.bashrc && \
    echo 'export PS1="\[\e[32;1m\]\u\[\e[m\]@\[\e[34;1m\]\H\[\e[m\]:\[\e[33;1m\]\w\[\e[m\]$ "' >> /home/vscode/.bashrc && \
    apt-get clean

COPY ./ /tmp/element-array-ephys/

RUN \
    # pipeline dependencies
    apt-get install gcc g++ ffmpeg libsm6 libxext6 -y && \
    pip install --no-cache-dir -e /tmp/element-array-ephys[elements,tests] && \
    # clean up
    rm -rf /tmp/element-array-ephys && \
    apt-get clean

ENV DJ_HOST fakeservices.datajoint.io
ENV DJ_USER root
ENV DJ_PASS simple

ENV EPHYS_ROOT_DATA_DIR /workspaces/element-array-ephys/example_data
ENV DATABASE_PREFIX neuro_

USER vscode

CMD bash -c "sudo rm /var/run/docker.pid; sudo dockerd"
