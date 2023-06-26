def test_generate_pipeline(pipeline):
    subject = pipeline["subject"]
    session = pipeline["session"]
    ephys = pipeline["ephys"]
    probe = pipeline["probe"]
    ephys_report = pipeline["ephys_report"]

    # test elements connection from lab, subject to Session
    assert subject.Subject.full_table_name in session.Session.parents()

    # test elements connection from Session to probe, ephys, ephys_report
    assert session.Session.full_table_name in ephys.ProbeInsertion.parents()
    assert probe.Probe.full_table_name in ephys.ProbeInsertion.parents()
    assert "spike_times" in (ephys.CuratedClustering.Unit.heading.secondary_attributes)

    assert all(
        [
            ephys.CuratedClustering.full_table_name
            in ephys_report.ProbeLevelReport.parents(),
            ephys.CuratedClustering.Unit.full_table_name
            in ephys_report.UnitLevelReport.parents(),
        ]
    )
    
    # test the connection between quality metric tables
    assert ephys.QualityMetrics.full_table_name in ephys_report.QualityMetricSet.parents()