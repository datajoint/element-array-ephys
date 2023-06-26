FROM datajoint/djbase:py3.9-debian-8eb1715

USER anaconda:anaconda

COPY ./workflow-array-ephys/docker/apt_requirements.txt /tmp/
RUN /entrypoint.sh echo "Installed dependencies."

WORKDIR /main/workflow-array-ephys

# Always move local - conditional install in setup.sh
COPY --chown=anaconda:anaconda ./element-lab/ /main/element-lab/
COPY --chown=anaconda:anaconda ./element-animal/ /main/element-animal/
COPY --chown=anaconda:anaconda ./element-session/ /main/element-session/
COPY --chown=anaconda:anaconda ./element-event/ /main/element-event/
COPY --chown=anaconda:anaconda ./element-interface/ /main/element-interface/
COPY --chown=anaconda:anaconda ./element-array-ephys/ /main/element-array-ephys/
COPY --chown=anaconda:anaconda ./workflow-array-ephys/ /main/workflow-array-ephys/

# Conditional install - local-all, local-dlc, or git
COPY --chown=anaconda:anaconda ./workflow-array-ephys/docker/setup.sh /main/
COPY --chown=anaconda:anaconda ./workflow-array-ephys/docker/.env /main/
RUN chmod 755 /main/setup.sh
RUN chmod 755 /main/.env
RUN /main/setup.sh
RUN rm -f ./dj_local_conf.json
