#! /bin/bash
alias ll='ls -aGg'
export $(grep -v '^#' /main/.env | xargs)

cd /main/
echo "INSALL OPTION:" $INSTALL_OPTION

# Always get djarchive
pip install --no-deps git+https://github.com/datajoint/djarchive-client.git

if [ "$INSTALL_OPTION" == "local-all" ]; then # all local installs, mapped from host
    for f in lab animal session interface; do
        pip install -e ./element-${f}
    done
    pip install -e ./element-array-ephys[nwb]
    pip install -e ./workflow-array-ephys
else  # all except this repo pip installed
    for f in lab animal session interface; do
         pip install git+https://github.com/${GITHUB_USERNAME}/element-${f}.git
    done
    if [ "$INSTALL_OPTION" == "local-ephys" ]; then # only array-ephys items from local
        pip install -e ./element-array-ephys[nwb]
        pip install -e ./workflow-array-ephys
    elif [ "$INSTALL_OPTION" == "git" ]; then # all from github
        pip install git+https://github.com/${GITHUB_USERNAME}/element-array-ephys.git
        pip install git+https://github.com/${GITHUB_USERNAME}/workflow-array-ephys.git
    fi
fi

# If test cmd contains pytest, install
if [[ "$TEST_CMD" == *pytest* ]]; then
    pip install pytest
    pip install pytest-cov
    pip install opencv-python
fi
