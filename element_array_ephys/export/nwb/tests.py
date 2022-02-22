from element_array_ephys.export.nwb.nwb import ecephys_session_to_nwb, write_nwb

import numpy as np
from pynwb.ecephys import ElectricalSeries


def test_convert_to_nwb():

    nwbfile = ecephys_session_to_nwb(
        dict(subject="subject5", session_datetime="2020-05-12 04:13:07")
    )

    for x in ("262716621", "714000838"):
        assert x in nwbfile.devices

    assert len(nwbfile.electrodes) == 1920
    for col in ("shank", "shank_row", "shank_col"):
        assert col in nwbfile.electrodes

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.acquisition[es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 2.34375e-06

    # make sure the ElectricalSeries objects don't share electrodes
    assert not set(nwbfile.acquisition["ElectricalSeries1"].electrodes.data) & set(
        nwbfile.acquisition["ElectricalSeries2"].electrodes.data
    )

    assert len(nwbfile.units) == 499
    for col in ("cluster_quality_label", "spike_depths"):
        assert col in nwbfile.units

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.processing["ecephys"].data_interfaces["LFP"][es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 4.6875e-06
        assert es.rate == 2500.0


def test_convert_to_nwb_with_dj_lfp():
    nwbfile = ecephys_session_to_nwb(
        dict(subject="subject5", session_datetime="2020-05-12 04:13:07"),
        lfp="dj",
        spikes=False,
    )

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.processing["ecephys"].data_interfaces["LFP"][es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 1.0
        assert isinstance(es.timestamps, np.ndarray)

