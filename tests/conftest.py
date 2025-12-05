import os
import pathlib

import datajoint as dj
import pytest

logger = dj.logger
_tear_down = True

# ---------------------- FIXTURES ----------------------


@pytest.fixture(autouse=True, scope="session")
def dj_config():
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
        }
    )
    os.environ["DATABASE_PREFIX"] = "test_"
    return


@pytest.fixture(autouse=True, scope="session")
def pipeline():
    from . import tutorial_pipeline as pipeline

    yield {
        "lab": pipeline.lab,
        "subject": pipeline.subject,
        "session": pipeline.session,
        "probe": pipeline.probe,
        "ephys": pipeline.ephys,
        "ephys_report": pipeline.ephys_report,
        "get_ephys_root_data_dir": pipeline.get_ephys_root_data_dir,
    }

    if _tear_down:
        pipeline.ephys_report.schema.drop()
        pipeline.ephys.schema.drop()
        pipeline.probe.schema.drop()
        pipeline.session.schema.drop()
        pipeline.subject.schema.drop()
        pipeline.lab.schema.drop()


@pytest.fixture(scope="session")
def insert_upstreams(pipeline):

    subject = pipeline["subject"]
    session = pipeline["session"]
    probe = pipeline["probe"]
    ephys = pipeline["ephys"]

    subject.Subject.insert1(
        dict(subject="subject5", subject_birth_date="2023-01-01", sex="U"),
        skip_duplicates=True,
    )

    session_key = dict(subject="subject5", session_datetime="2023-01-01 00:00:00")
    session.Session.insert1(session_key, skip_duplicates=True)
    session_dir = "raw/subject5/session1"

    session.SessionDirectory.insert1(
        dict(**session_key, session_dir=session_dir), skip_duplicates=True
    )
    probe.Probe.insert1(
        dict(probe="714000838", probe_type="neuropixels 1.0 - 3B"), skip_duplicates=True
    )
    ephys.ProbeInsertion.insert1(
        dict(
            **session_key,
            insertion_number=1,
            probe="714000838",
        ),
        skip_duplicates=True,
    )

    return


@pytest.fixture(scope="session")
def populate_ephys_recording(pipeline, insert_upstreams):
    ephys = pipeline["ephys"]
    ephys.EphysRecording.populate()

    return


@pytest.fixture(scope="session")
def populate_lfp(pipeline, insert_upstreams):
    ephys = pipeline["ephys"]
    ephys.LFP.populate()

    return


@pytest.fixture(scope="session")
def insert_clustering_task(pipeline, populate_ephys_recording):
    ephys = pipeline["ephys"]
    params = {}
    params["SI_PREPROCESSING_METHOD"] = "CatGT"
    params["SI_SORTING_PARAMS"] = {
        "minfr_goodchannels": 0.1,
        "lam": 10,
        "AUCsplit": 0.9,
        "minFR": 0.02,
        "momentum": [20, 400],
        "sigmaMask": 30,
        "nfilt_factor": 4,
        "ntbuff": 64,
        "scaleproc": 200,
        "nPCs": 3,
    }
    params["SI_POSTPROCESSING_PARAMS"] = {
        "extensions": {
            "random_spikes": {},
            "waveforms": {},
            "templates": {},
            "noise_levels": {},
            # "amplitude_scalings": {},
            "correlograms": {},
            "isi_histograms": {},
            "principal_components": {"n_components": 5, "mode": "by_channel_local"},
            "spike_amplitudes": {},
            "spike_locations": {},
            "template_metrics": {"include_multi_channel_metrics": True},
            "template_similarity": {},
            "unit_locations": {},
            "quality_metrics": {},
        },
        "job_kwargs": {"n_jobs": 0.8, "chunk_duration": "1s"},
        "export_to_phy": True,
        "export_report": True,
    }

    ephys.ClusteringParamSet.insert_new_params(
        clustering_method="kilosort2.5",
        paramset_desc="Default parameter set for Kilosort2.0 with SpikeInterface",
        params=params,
        paramset_idx=0,
    )

    session_key = dict(subject="subject5", session_datetime="2023-01-01 00:00:00")

    ephys.ClusteringTask.insert1(
        dict(
            session_key,
            insertion_number=1,
            paramset_idx=0,
            task_mode="load",  # load or trigger
            clustering_output_dir="processed/subject5/session1/probe_1/kilosort2-5_1",
        ),
        skip_duplicates=True,
    )

    return


@pytest.fixture(scope="session")
def processing(pipeline, insert_clustering_task):

    ephys = pipeline["ephys"]
    ephys.Clustering.populate()
    ephys.CuratedClustering.populate()
    ephys.WaveformSet.populate()
    ephys.QualityMetrics.populate()

    return
