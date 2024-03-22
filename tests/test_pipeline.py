import numpy as np
import pandas as pd
import datetime
from uuid import UUID


def test_generate_pipeline(pipeline):
    subject = pipeline["subject"]
    session = pipeline["session"]
    ephys = pipeline["ephys"]
    probe = pipeline["probe"]

    # test elements connection from lab, subject to Session
    assert subject.Subject.full_table_name in session.Session.parents()

    # test elements connection from Session to probe, ephys, ephys_report
    assert session.Session.full_table_name in ephys.ProbeInsertion.parents()
    assert probe.Probe.full_table_name in ephys.ProbeInsertion.parents()
    assert "spike_times" in (ephys.CuratedClustering.Unit.heading.secondary_attributes)


def test_insert_upstreams(pipeline, insert_upstreams):
    """Check number of subjects inserted into the `subject.Subject` table"""
    subject = pipeline["subject"]
    session = pipeline["session"]
    probe = pipeline["probe"]
    ephys = pipeline["ephys"]

    assert len(subject.Subject()) == 1
    assert len(session.Session()) == 1
    assert len(probe.Probe()) == 1
    assert len(ephys.ProbeInsertion()) == 1


def test_populate_ephys_recording(pipeline, populate_ephys_recording):
    ephys = pipeline["ephys"]

    assert ephys.EphysRecording.fetch1() == {
        "subject": "subject5",
        "session_datetime": datetime.datetime(2023, 1, 1, 0, 0),
        "insertion_number": 1,
        "electrode_config_hash": UUID("8d4cc6d8-a02d-42c8-bf27-7459c39ea0ee"),
        "acq_software": "SpikeGLX",
        "sampling_rate": 30000.0,
        "recording_datetime": datetime.datetime(2018, 7, 3, 20, 32, 28),
        "recording_duration": 338.666,
    }
    assert (
        ephys.EphysRecording.EphysFile.fetch1("file_path")
        == "raw/subject5/session1/probe_1/npx_g0_t0.imec.ap.meta"
    )


def test_populate_lfp(pipeline, populate_lfp):
    ephys = pipeline["ephys"]

    assert np.mean(ephys.LFP.fetch1("lfp_mean")) == -716.0220556825378
    assert len((ephys.LFP.Electrode).fetch("electrode")) == 43


def test_insert_clustering_task(pipeline, insert_clustering_task):
    ephys = pipeline["ephys"]

    assert ephys.ClusteringParamSet.fetch1("param_set_hash") == UUID(
        "de78cee1-526f-319e-b6d5-8a2ba04963d8"
    )

    assert ephys.ClusteringTask.fetch1() == {
        "subject": "subject5",
        "session_datetime": datetime.datetime(2023, 1, 1, 0, 0),
        "insertion_number": 1,
        "paramset_idx": 0,
        "clustering_output_dir": "processed/subject5/session1/probe_1/kilosort2-5_1",
        "task_mode": "load",
    }


def test_processing(pipeline, processing):

    ephys = pipeline["ephys"]

    # test ephys.CuratedClustering
    assert len(ephys.CuratedClustering.Unit & 'cluster_quality_label = "good"') == 176
    assert np.sum(ephys.CuratedClustering.Unit.fetch("spike_count")) == 328167
    # test ephys.WaveformSet
    waveforms = np.vstack(
        (ephys.WaveformSet.PeakWaveform).fetch("peak_electrode_waveform")
    )
    assert waveforms.shape == (227, 82)

    # test ephys.QualityMetrics
    cluster_df = (ephys.QualityMetrics.Cluster).fetch(format="frame", order_by="unit")
    waveform_df = (ephys.QualityMetrics.Waveform).fetch(format="frame", order_by="unit")
    test_df = pd.concat([cluster_df, waveform_df], axis=1).reset_index()
    test_value = test_df.select_dtypes(include=[np.number]).mean().values

    assert np.allclose(
        test_value,
        np.array(
            [
                1.00000000e00,
                0.00000000e00,
                1.13000000e02,
                4.26880089e00,
                1.24162431e00,
                7.17929515e-01,
                4.41633793e-01,
                3.08736082e-01,
                1.24039274e15,
                1.66763828e-02,
                4.33231948e00,
                7.12304747e-01,
                1.48995215e-02,
                7.73432472e-02,
                5.06451613e00,
                7.79528634e00,
                6.30182452e-01,
                1.19562726e02,
                7.90175419e-01,
                np.nan,
                8.78436780e-01,
                1.08028193e-01,
                -5.19418717e-02,
                2.36035242e02,
                7.48443665e-02,
                2.77550214e-02,
            ]
        ),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
