import os
import pathlib
import sys

from element_interface.utils import find_full_path, find_root_directory

docker_root = "/main/test_data/workflow_ephys_data1"


def test_ingest_subjects(pipeline, ingest_data):
    """Check number of subjects inserted into the `subject.Subject` table"""
    subject = pipeline["subject"]
    assert len(subject.Subject()) == 6


def test_ingest_sessions(pipeline, ingest_data):
    ephys = pipeline["ephys"]
    probe = pipeline["probe"]
    session = pipeline["session"]

    assert len(session.Session()) == 7
    assert len(probe.Probe()) == 9
    assert len(ephys.ProbeInsertion()) == 13

    session_info = ingest_data["sessions.csv"]["content"][1].split(",")

    assert (session.SessionDirectory & {"subject": session_info[0]}).fetch1(
        "session_dir"
    ) == session_info[1]


def test_find_valid_full_path(pipeline, ingest_data):

    if not os.environ.get("IS_DOCKER", False):
        return  # It doesn't make sense to assert the root testing locally

    get_ephys_root_data_dir = pipeline["get_ephys_root_data_dir"]
    ephys_root_data_dir = (
        [get_ephys_root_data_dir()]
        if not isinstance(get_ephys_root_data_dir(), list)
        else get_ephys_root_data_dir()
    )

    # add more options for root directories
    if sys.platform == "win32":  # win32 even if Windows 64-bit
        ephys_root_data_dir = ephys_root_data_dir + ["J:/", "M:/"]
    else:
        ephys_root_data_dir = ephys_root_data_dir + ["mnt/j", "mnt/m"]

    # test: providing relative-path: correctly search for the full-path
    session_info = ingest_data["sessions.csv"]["content"][1].split(",")

    session_full_path = find_full_path(ephys_root_data_dir, session_info[1])

    full_path = pathlib.Path(docker_root, "subject1/session1")

    assert full_path == session_full_path, str(
        "Session path does not match docker root:"
        + f"\n\t{full_path}\n\t{session_full_path}"
    )


def test_find_root_directory(pipeline, ingest_data):
    """
    Test that `find_root_directory` works correctly.
    """

    get_ephys_root_data_dir = pipeline["get_ephys_root_data_dir"]
    ephys_root_data_dir = (
        [get_ephys_root_data_dir()]
        if not isinstance(get_ephys_root_data_dir(), list)
        else get_ephys_root_data_dir()
    )
    # add more options for root directories
    if sys.platform == "win32":
        ephys_root_data_dir = ephys_root_data_dir + ["J:/", "M:/"]
    else:
        ephys_root_data_dir = ephys_root_data_dir + ["mnt/j", "mnt/m"]

    ephys_root_data_dir = [pathlib.Path(p) for p in ephys_root_data_dir]

    # test: providing full-path: correctly search for the root_dir
    session_info = ingest_data["sessions.csv"]["content"][1].split(",")

    if os.environ.get("IS_DOCKER", False):
        session_full_path = pathlib.Path(docker_root, session_info[1])
        root_dir = find_root_directory(ephys_root_data_dir, session_full_path)
        assert (
            root_dir.as_posix() == docker_root
        ), f"Root path does not match: {docker_root}"
    else:
        session_full_path = find_full_path(get_ephys_root_data_dir(), session_info[1])
        root_dir = find_root_directory(ephys_root_data_dir, session_full_path)
        assert root_dir in ephys_root_data_dir, "Problems finding root dir"


def test_paramset_insert(kilosort_paramset, pipeline):
    ephys = pipeline["ephys"]
    from element_interface.utils import dict_to_uuid

    method, desc, paramset_hash = (
        ephys.ClusteringParamSet & {"paramset_idx": 0}
    ).fetch1("clustering_method", "paramset_desc", "param_set_hash")
    assert method == "kilosort2.5"
    assert desc == "Spike sorting using Kilosort2.5"
    assert (
        dict_to_uuid({**kilosort_paramset, "clustering_method": method})
        == paramset_hash
    )
