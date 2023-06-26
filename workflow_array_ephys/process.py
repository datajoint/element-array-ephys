from workflow_array_ephys.pipeline import ephys


def run(
    display_progress: bool = True,
    reserve_jobs: bool = False,
    suppress_errors: bool = False,
):
    """Execute all populate commands in Element Array Ephys

    Args:
        display_progress (bool, optional): See DataJoint `populate`. Defaults to True.
        reserve_jobs (bool, optional): See DataJoint `populate`. Defaults to False.
        suppress_errors (bool, optional): See DataJoint `populate`. Defaults to False.
    """

    populate_settings = {
        "display_progress": display_progress,
        "reserve_jobs": reserve_jobs,
        "suppress_errors": suppress_errors,
    }

    print("\n---- Populate ephys.EphysRecording ----")
    ephys.EphysRecording.populate(**populate_settings)

    print("\n---- Populate ephys.LFP ----")
    ephys.LFP.populate(**populate_settings)

    print("\n---- Populate ephys.Clustering ----")
    ephys.Clustering.populate(**populate_settings)

    print("\n---- Populate ephys.CuratedClustering ----")
    ephys.CuratedClustering.populate(**populate_settings)

    print("\n---- Populate ephys.WaveformSet ----")
    ephys.WaveformSet.populate(**populate_settings)


if __name__ == "__main__":
    run()
