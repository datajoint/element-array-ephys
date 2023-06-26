import logging
import os
import pathlib
from contextlib import nullcontext
from pathlib import Path

import datajoint as dj
import pytest
from element_interface.utils import QuietStdOut, find_full_path, value_to_bool

from workflow_array_ephys.ingest import ingest_lab, ingest_sessions, ingest_subjects
from workflow_array_ephys.paths import get_ephys_root_data_dir

# ------------------- SOME CONSTANTS -------------------


logger = logging.getLogger("datajoint")

pathlib.Path("./tests/user_data").mkdir(exist_ok=True)
pathlib.Path("./tests/user_data/lab").mkdir(exist_ok=True)

sessions_dirs = [
    "subject1/session1",
    "subject2/session1",
    "subject2/session2",
    "subject3/session1",
    "subject4/experiment1",
    "subject5/session1",
    "subject6/session1",
]


def pytest_addoption(parser):
    """
    Permit constants when calling pytest at command line e.g., pytest --dj-verbose False

    Arguments:
        --dj-verbose (bool):  Default True. Pass print statements from Elements.
        --dj-teardown (bool): Default True. Delete pipeline on close.
        --dj-datadir (str):  Default ./tests/user_data. Relative path of test CSV data.
    """
    parser.addoption(
        "--dj-verbose",
        action="store",
        default="True",
        help="Verbose for dj items: True or False",
        choices=("True", "False"),
    )
    parser.addoption(
        "--dj-teardown",
        action="store",
        default="True",
        help="Verbose for dj items: True or False",
        choices=("True", "False"),
    )
    parser.addoption(
        "--dj-datadir",
        action="store",
        default="./tests/user_data",
        help="Relative path for saving tests data",
    )


@pytest.fixture(autouse=True, scope="session")
def setup(request):
    """Take passed commandline variables, set as global"""
    global verbose, _tear_down, test_user_data_dir, verbose_context

    verbose = value_to_bool(request.config.getoption("--dj-verbose"))
    _tear_down = value_to_bool(request.config.getoption("--dj-teardown"))
    test_user_data_dir = Path(request.config.getoption("--dj-datadir"))
    test_user_data_dir.mkdir(exist_ok=True)

    if not verbose:
        logging.getLogger("datajoint").setLevel(logging.CRITICAL)

    verbose_context = nullcontext() if verbose else QuietStdOut()

    yield verbose_context, verbose


# --------------------  HELPER CLASS --------------------


def null_function(*args, **kwargs):
    pass


# ---------------------- FIXTURES ----------------------


@pytest.fixture(autouse=True, scope="session")
def dj_config(setup):
    """If dj_local_config exists, load"""
    if pathlib.Path("./dj_local_conf.json").exists():
        dj.config.load("./dj_local_conf.json")
    dj.config.update(
        {
            "safemode": False,
            "database.host": os.environ.get("DJ_HOST") or dj.config["database.host"],
            "database.password": os.environ.get("DJ_PASS")
            or dj.config["database.password"],
            "database.user": os.environ.get("DJ_USER") or dj.config["database.user"],
            "custom": {
                "ephys_mode": (
                    os.environ.get("EPHYS_MODE") or dj.config["custom"]["ephys_mode"]
                ),
                "database.prefix": (
                    os.environ.get("DATABASE_PREFIX")
                    or dj.config["custom"]["database.prefix"]
                ),
                "ephys_root_data_dir": (
                    os.environ.get("EPHYS_ROOT_DATA_DIR").split(",")
                    if os.environ.get("EPHYS_ROOT_DATA_DIR")
                    else dj.config["custom"]["ephys_root_data_dir"]
                ),
            },
        }
    )
    return


@pytest.fixture(scope="session")
def test_data(dj_config):
    """If data does not exist or partial data is present,
    attempt download with DJArchive to the first listed root directory"""
    test_data_exists = True

    for p in sessions_dirs:
        try:
            find_full_path(get_ephys_root_data_dir(), p)
        except FileNotFoundError:
            test_data_exists = False  # If data not found
            break

    if not test_data_exists:  # attempt to djArchive dowload
        try:
            dj.config["custom"].update(
                {
                    "djarchive.client.endpoint": os.environ[
                        "DJARCHIVE_CLIENT_ENDPOINT"
                    ],
                    "djarchive.client.bucket": os.environ["DJARCHIVE_CLIENT_BUCKET"],
                    "djarchive.client.access_key": os.environ[
                        "DJARCHIVE_CLIENT_ACCESSKEY"
                    ],
                    "djarchive.client.secret_key": os.environ[
                        "DJARCHIVE_CLIENT_SECRETKEY"
                    ],
                }
            )
        except KeyError as e:
            raise FileNotFoundError(
                "Full test data not available.\nAttempting to download from DJArchive,"
                + " but no credentials found in environment variables.\nError:"
                + str(e)
            )

        import djarchive_client

        client = djarchive_client.client()

        test_data_dir = get_ephys_root_data_dir()
        if isinstance(test_data_dir, list):  # if multiple root dirs, first
            test_data_dir = test_data_dir[0]

        client.download(
            "workflow-array-ephys-benchmark",
            "v2",
            str(test_data_dir),
            create_target=False,
        )
    return


@pytest.fixture(autouse=True, scope="session")
def pipeline():
    from workflow_array_ephys import pipeline

    yield {
        "subject": pipeline.subject,
        "lab": pipeline.lab,
        "ephys": pipeline.ephys,
        "probe": pipeline.probe,
        "ephys_report": pipeline.ephys_report,
        "session": pipeline.session,
        "get_ephys_root_data_dir": pipeline.get_ephys_root_data_dir,
        "ephys_mode": pipeline.ephys_mode,
    }

    if _tear_down:
        with verbose_context:
            pipeline.subject.Subject.delete()


@pytest.fixture(scope="session")
def ingest_data(setup, pipeline, test_data):
    """For each input, generates csv in test_user_data_dir and ingests in schema"""
    # CSV as list of 3: filename, relevant tables, content
    all_csvs = {
        "lab/labs.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "lab,lab_name,organization,org_name,address,"
                + "time_zone,location,location_description",
                "LabA,The Example Lab,Uni1,Example Uni,'221B Baker St,London NW1 6XE,UK',"
                + "UTC+0,Example Building,'2nd floor lab dedicated to all fictional experiments.'",
                "LabB,The Other Lab,Uni2,Other Uni,'Oxford OX1 2JD, United Kingdom',"
                + "UTC+0,Other Building,'fictional campus dedicated to imaginaryexperiments.'",
            ],
        },
        "lab/projects.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "project,project_description,project_title,project_start_date,"
                + "repository_url,repository_name,codeurl",
                "ProjA,Example project to populate element-lab,"
                + "Example project to populate element-lab,2020-01-01,"
                + "https://github.com/datajoint/element-lab/,"
                + "element-lab,https://github.com/datajoint/element"
                + "-lab/tree/main/element_lab",
                "ProjB,Other example project to populate element-lab,"
                + "Other example project to populate element-lab,2020-01-02,"
                + "https://github.com/datajoint/element-session/,"
                + "element-session,https://github.com/datajoint/"
                + "element-session/tree/main/element_session",
            ],
        },
        "lab/project_users.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "user,project",
                "Sherlock,ProjA",
                "Sherlock,ProjB",
                "Watson,ProjB",
                "Dr. Candace Pert,ProjA",
                "User1,ProjA",
            ],
        },
        "lab/publications.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "project,publication",
                "ProjA,arXiv:1807.11104",
                "ProjA,arXiv:1807.11104v1",
            ],
        },
        "lab/keywords.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "project,keyword",
                "ProjA,Study",
                "ProjA,Example",
                "ProjB,Alternate",
            ],
        },
        "lab/protocols.csv": {
            "func": null_function,
            "args": {},
            "content": [
                "protocol,protocol_type,protocol_description",
                "ProtA,IRB expedited review,Protocol for managing " + "data ingestion",
                "ProtB,Alternative Method,Limited protocol for " + "piloting only",
            ],
        },
        "lab/users.csv": {
            "func": ingest_lab,
            "args": {
                "lab_csv_path": f"{test_user_data_dir}/lab/labs.csv",
                "project_csv_path": f"{test_user_data_dir}/lab/projects.csv",
                "publication_csv_path": f"{test_user_data_dir}/lab/publications.csv",
                "keyword_csv_path": f"{test_user_data_dir}/lab/keywords.csv",
                "protocol_csv_path": f"{test_user_data_dir}/lab/protocols.csv",
                "users_csv_path": f"{test_user_data_dir}/lab/users.csv",
                "project_user_csv_path": f"{test_user_data_dir}/lab/project_users.csv",
            },
            "content": [
                "lab,user,user_role,user_email,user_cellphone",
                "LabA,Sherlock,PI,Sherlock@BakerSt.com," + "+44 20 7946 0344",
                "LabA,Watson,Dr,DrWatson@BakerSt.com,+44 73 8389 1763",
                "LabB,Dr. Candace Pert,PI,Pert@gmail.com," + "+44 74 4046 5899",
                "LabA,User1,Lab Tech,fake@email.com,+44 1632 960103",
                "LabB,User2,Lab Tech,fake2@email.com,+44 1632 960102",
            ],
        },
        "subjects.csv": {
            "func": ingest_subjects,
            "args": {"subject_csv_path": f"{test_user_data_dir}/subjects.csv"},
            "content": [
                "subject,sex,subject_birth_date,subject_description",
                "subject1,F,2020-01-01 00:00:01,dl56",
                "subject2,M,2020-01-01 00:00:01,SC035",
                "subject3,M,2020-01-01 00:00:01,SC038",
                "subject4,M,2020-01-01 00:00:01,oe_talab",
                "subject5,F,2020-01-01 00:00:01,rich",
                "subject6,F,2020-01-01 00:00:01,manuel",
            ],
        },
        "sessions.csv": {
            "func": ingest_sessions,
            "args": {"session_csv_path": f"{test_user_data_dir}/sessions.csv"},
            "content": [
                "subject,session_dir,session_note,user",
                f"subject1,{sessions_dirs[0]},Data collection notes,User2",
                f"subject2,{sessions_dirs[1]},Data collection notes,User2",
                f"subject2,{sessions_dirs[2]},Interrupted session,User2",
                f"subject3,{sessions_dirs[3]},Data collection notes,User1",
                f"subject4,{sessions_dirs[4]},Successful data collection,User2",
                f"subject5,{sessions_dirs[5]},Successful data collection,User1",
                f"subject6,{sessions_dirs[6]},Ambient temp abnormally low,User2",
            ],
        },
    }
    # If data in last table, presume didn't tear down last time, skip insert
    if len(pipeline["ephys"].Clustering()) == 0:
        for csv_filename, csv_dict in all_csvs.items():
            csv_path = test_user_data_dir / csv_filename  # add prefix for rel path
            Path(csv_path).write_text("\n".join(csv_dict["content"]) + "\n")
            csv_dict["func"](verbose=verbose, skip_duplicates=True, **csv_dict["args"])

    yield all_csvs

    if _tear_down:
        with verbose_context:
            for csv in all_csvs:
                csv_path = test_user_data_dir / csv
                csv_path.unlink()


@pytest.fixture(scope="session")
def testdata_paths():
    """Paths for test data 'subjectX/sessionY/probeZ/etc'"""
    return {
        "npx3A-p1-ks": "subject5/session1/probe_1/kilosort2-5_1",
        "npx3A-p2-ks": "subject5/session1/probe_2/kilosort2-5_1",
        "oe_npx3B-ks": "subject4/experiment1/recording1/continuous/"
        + "Neuropix-PXI-100.0/ks",
        "sglx_npx3A-p1": "subject5/session1/probe_1",
        "oe_npx3B": "subject4/experiment1/recording1/continuous/"
        + "Neuropix-PXI-100.0",
        "sglx_npx3B-p1": "subject6/session1/towersTask_g0_imec0",
        "npx3B-p1-ks": "subject6/session1/towersTask_g0_imec0",
    }


@pytest.fixture(scope="session")
def ephys_insertionlocation(pipeline, ingest_data):
    """Insert probe location into ephys.InsertionLocation"""
    ephys = pipeline["ephys"]

    for probe_insertion_key in ephys.ProbeInsertion.fetch("KEY"):
        ephys.InsertionLocation.insert1(
            dict(
                **probe_insertion_key,
                skull_reference="Bregma",
                ap_location=0,
                ml_location=0,
                depth=0,
                theta=0,
                phi=0,
                beta=0,
            ),
            skip_duplicates=True,
        )
    yield

    if _tear_down:
        with verbose_context:
            ephys.InsertionLocation.delete()


@pytest.fixture(scope="session")
def kilosort_paramset(pipeline):
    """Insert kilosort parameters into ephys.ClusteringParamset"""
    ephys = pipeline["ephys"]

    params_ks = {
        "fs": 30000,
        "fshigh": 150,
        "minfr_goodchannels": 0.1,
        "Th": [10, 4],
        "lam": 10,
        "AUCsplit": 0.9,
        "minFR": 0.02,
        "momentum": [20, 400],
        "sigmaMask": 30,
        "ThPr": 8,
        "spkTh": -6,
        "reorder": 1,
        "nskip": 25,
        "GPU": 1,
        "Nfilt": 1024,
        "nfilt_factor": 4,
        "ntbuff": 64,
        "whiteningRange": 32,
        "nSkipCov": 25,
        "scaleproc": 200,
        "nPCs": 3,
        "useRAM": 0,
    }

    # Insert here, since most of the test will require this paramset inserted
    ephys.ClusteringParamSet.insert_new_params(
        clustering_method="kilosort2.5",
        paramset_desc="Spike sorting using Kilosort2.5",
        params=params_ks,
        paramset_idx=0,
    )

    yield params_ks

    if _tear_down:
        with verbose_context:
            (ephys.ClusteringParamSet & "paramset_idx = 0").delete()


@pytest.fixture(scope="session")
def ephys_recordings(pipeline, ingest_data):
    """Populate ephys.EphysRecording"""
    ephys = pipeline["ephys"]

    ephys.EphysRecording.populate()

    yield

    if _tear_down:
        with verbose_context:
            ephys.EphysRecording.delete()


@pytest.fixture(scope="session")
def clustering_tasks(pipeline, kilosort_paramset, ephys_recordings):
    """Insert keys from ephys.EphysRecording into ephys.Clustering"""
    ephys = pipeline["ephys"]

    for ephys_rec_key in (ephys.EphysRecording - ephys.ClusteringTask).fetch("KEY"):
        ephys_file_path = pathlib.Path(
            ((ephys.EphysRecording.EphysFile & ephys_rec_key).fetch("file_path"))[0]
        )
        ephys_file = find_full_path(get_ephys_root_data_dir(), ephys_file_path)
        recording_dir = ephys_file.parent
        kilosort_dir = next(recording_dir.rglob("spike_times.npy")).parent
        ephys.ClusteringTask.insert1(
            {
                **ephys_rec_key,
                "paramset_idx": 0,
                "task_mode": "load",
                "clustering_output_dir": kilosort_dir.as_posix(),
            },
            skip_duplicates=True,
        )

    yield

    if _tear_down:
        with verbose_context:
            ephys.ClusteringTask.delete()


@pytest.fixture(scope="session")
def clustering(clustering_tasks, pipeline):
    """Populate ephys.Clustering"""
    ephys = pipeline["ephys"]

    ephys.Clustering.populate()

    yield

    if _tear_down:
        with verbose_context:
            ephys.Clustering.delete()


@pytest.fixture(scope="session")
def curations(clustering, pipeline):
    """Insert keys from ephys.ClusteringTask into ephys.Curation"""
    ephys_mode = pipeline["ephys_mode"]

    if ephys_mode == "no-curation":
        yield
    else:
        ephys = pipeline["ephys"]

        for key in (ephys.ClusteringTask - ephys.Curation).fetch("KEY"):
            ephys.Curation().create1_from_clustering_task(key)

        yield

        if _tear_down:
            with verbose_context:
                ephys.Curation.delete()
