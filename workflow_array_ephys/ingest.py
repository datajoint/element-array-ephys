import csv
import logging
import re

from element_array_ephys.readers import openephys, spikeglx
from element_interface.utils import (
    find_full_path,
    find_root_directory,
    ingest_csv_to_table,
)

from workflow_array_ephys.paths import get_ephys_root_data_dir
from workflow_array_ephys.pipeline import (
    ephys,
    event,
    lab,
    probe,
    project,
    session,
    subject,
    trial,
)

logger = logging.getLogger("datajoint")


def ingest_lab(
    lab_csv_path="./user_data/lab/labs.csv",
    project_csv_path="./user_data/lab/projects.csv",
    publication_csv_path="./user_data/lab/publications.csv",
    keyword_csv_path="./user_data/lab/keywords.csv",
    protocol_csv_path="./user_data/lab/protocols.csv",
    users_csv_path="./user_data/lab/users.csv",
    project_user_csv_path="./user_data/lab/project_users.csv",
    skip_duplicates=True,
    verbose=True,
):
    """Inserts data from a CSVs into their corresponding lab schema tables.

    By default, uses data from workflow/user_data/lab/

    Args:
        lab_csv_path (str): relative path of lab csv
        project_csv_path (str): relative path of project csv
        publication_csv_path (str): relative path of publication csv
        keyword_csv_path (str): relative path of keyword csv
        protocol_csv_path (str): relative path of protocol csv
        users_csv_path (str): relative path of users csv
        project_user_csv_path (str): relative path of project users csv
        skip_duplicates=True (str): datajoint insert function param
        verbose (str): print number inserted (i.e., table length change)
    """

    # List with repeats for when multiple dj.tables fed by same CSV
    csvs = [
        lab_csv_path,
        lab_csv_path,
        lab_csv_path,
        lab_csv_path,
        project_csv_path,
        project_csv_path,
        publication_csv_path,
        keyword_csv_path,
        protocol_csv_path,
        protocol_csv_path,
        users_csv_path,
        users_csv_path,
        users_csv_path,
        project_user_csv_path,
    ]
    tables = [
        lab.Organization(),
        lab.Lab(),
        lab.Lab.Organization(),
        lab.Location(),
        lab.Project(),
        lab.ProjectSourceCode(),
        lab.ProjectPublication(),
        lab.ProjectKeywords(),
        lab.ProtocolType(),
        lab.Protocol(),
        lab.UserRole(),
        lab.User(),
        lab.LabMembership(),
        lab.ProjectUser(),
    ]

    ingest_csv_to_table(csvs, tables, skip_duplicates=skip_duplicates, verbose=verbose)

    # For project schema
    project_csvs = [
        project_csv_path,
        project_user_csv_path,
        keyword_csv_path,
        publication_csv_path,
        project_csv_path,
    ]

    project_tables = [
        project.Project(),
        project.ProjectPersonnel(),
        project.ProjectKeywords(),
        project.ProjectPublication(),
        project.ProjectSourceCode(),
    ]

    ingest_csv_to_table(
        project_csvs, project_tables, skip_duplicates=skip_duplicates, verbose=verbose
    )


def ingest_subjects(
    subject_csv_path: str = "./user_data/subjects.csv",
    skip_duplicates: bool = True,
    verbose: bool = True,
):
    """Ingest subjects listed in the subject column of ./user_data/subjects.csv

    Args:
        subject_csv_path (str, optional): Relative path to subject csv.
            Defaults to "./user_data/subjects.csv".
        skip_duplicates (bool, optional): See DataJoint `insert` function. Default True.
        verbose (bool, optional): Print number inserted (i.e., table length change).
            Defaults to True.
    """
    csvs = [subject_csv_path]
    tables = [subject.Subject()]

    ingest_csv_to_table(csvs, tables, skip_duplicates=skip_duplicates, verbose=verbose)


def ingest_sessions(
    session_csv_path: str = "./user_data/sessions.csv", verbose: bool = True, **_
):
    """Ingest SpikeGLX and OpenEphys files from directories listed in csv

    Args:
        session_csv_path (str, optional): List of sessions.
            Defaults to "./user_data/sessions.csv".
        verbose (bool, optional): Print number inserted (i.e., table length change).
            Defaults to True.

    Raises:
        FileNotFoundError: Neither SpikeGLX nor OpenEphys recording files found in dir
        NotImplementedError: Acquisition software provided does not match those
            implemented - SpikeGLX and OpenEphys
    """

    # ---------- Insert new "Session" and "ProbeInsertion" ---------
    with open(session_csv_path, newline="") as f:
        input_sessions = list(csv.DictReader(f, delimiter=","))

    # Folder structure: root / subject / session / probe / .ap.meta
    (session_list, session_dir_list) = ([], [])
    session_note_list, session_experimenter_list, lab_user_list = [], [], []
    probe_list, probe_insertion_list = [], []

    for this_session in input_sessions:
        session_dir = find_full_path(
            get_ephys_root_data_dir(), this_session["session_dir"]
        )
        session_datetimes, insertions = [], []

        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in zip(
            ["*.ap.meta", "*.oebin"], ["SpikeGLX", "OpenEphys"]
        ):
            ephys_meta_filepaths = [fp for fp in session_dir.rglob(ephys_pattern)]
            if len(ephys_meta_filepaths):
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                "Ephys recording data not found! Neither SpikeGLX "
                + "nor OpenEphys recording files found in: "
                + f"{session_dir}"
            )

        if acq_software == "SpikeGLX":
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)

                probe_key = {
                    "probe_type": spikeglx_meta.probe_model,
                    "probe": spikeglx_meta.probe_SN,
                }
                if (
                    probe_key["probe"] not in [p["probe"] for p in probe_list]
                    and probe_key not in probe.Probe()
                ):
                    probe_list.append(probe_key)

                probe_dir = meta_filepath.parent
                probe_number = re.search("(imec)?\d{1}$", probe_dir.name).group()
                probe_number = int(probe_number.replace("imec", ""))

                insertions.append(
                    {
                        "probe": spikeglx_meta.probe_SN,
                        "insertion_number": int(probe_number),
                    }
                )
                session_datetimes.append(spikeglx_meta.recording_time)
        elif acq_software == "OpenEphys":
            loaded_oe = openephys.OpenEphys(session_dir)
            session_datetimes.append(loaded_oe.experiment.datetime)
            for probe_idx, oe_probe in enumerate(loaded_oe.probes.values()):
                probe_key = {
                    "probe_type": oe_probe.probe_model,
                    "probe": oe_probe.probe_SN,
                }
                if (
                    probe_key["probe"] not in [p["probe"] for p in probe_list]
                    and probe_key not in probe.Probe()
                ):
                    probe_list.append(probe_key)
                insertions.append(
                    {"probe": oe_probe.probe_SN, "insertion_number": probe_idx}
                )
        else:
            raise NotImplementedError(
                "Unknown acquisition software: " + f"{acq_software}"
            )

        # new session/probe-insertion
        session_key = {
            "subject": this_session["subject"],
            "session_datetime": min(session_datetimes),
        }
        if session_key not in session.Session():
            session_list.append(session_key)
            root_dir = find_root_directory(get_ephys_root_data_dir(), session_dir)
            session_dir_list.append(
                {
                    **session_key,
                    "session_dir": session_dir.relative_to(root_dir).as_posix(),
                }
            )
            session_note_list.append(
                {**session_key, "session_note": this_session["session_note"]}
            )
            session_experimenter_list.append(
                {**session_key, "user": this_session["user"]}
            )
            lab_user_list.append(
                (this_session["user"], "", "", "")
            )  # empty email/phone/name
            probe_insertion_list.extend(
                [{**session_key, **insertion} for insertion in insertions]
            )

    session.Session.insert(session_list)
    lab.User.insert(lab_user_list, skip_duplicates=True)
    session.SessionDirectory.insert(session_dir_list)
    session.SessionNote.insert(session_note_list)
    session.SessionExperimenter.insert(session_experimenter_list)

    log_string = "---- Inserting %d entry(s) into %s ----"

    if verbose:
        logger.info(log_string % (len(session_list), "session.Session"))

    probe.Probe.insert(probe_list)
    if verbose:
        logger.info(log_string % (len(probe_list), "probe.Probe"))

    ephys.ProbeInsertion.insert(probe_insertion_list)
    if verbose:
        logger.info(log_string % (len(probe_insertion_list), "ephys.ProbeInsertion"))
        logger.info("---- Successfully completed ingest_subjects ----")


def ingest_events(
    recording_csv_path: str = "./user_data/behavior_recordings.csv",
    block_csv_path: str = "./user_data/blocks.csv",
    trial_csv_path: str = "./user_data/trials.csv",
    event_csv_path: str = "./user_data/events.csv",
    skip_duplicates: bool = True,
    verbose: bool = True,
):
    """Ingest each level of experiment hierarchy for element-trial

    Ingestion hierarchy includes:
        recording, block (i.e., phases of trials), trials (repeated units),
        events (optionally 0-duration occurrences within trial).

    Note: This ingestion function is duplicated across wf-array-ephys and wf-calcium-imaging

    Args:
        recording_csv_path (str, optional): Relative path to recording csv.
            Defaults to "./user_data/behavior_recordings.csv".
        block_csv_path (str, optional): Relative path to block csv.
            Defaults to "./user_data/blocks.csv".
        trial_csv_path (str, optional): Relative path to trial csv.
            Defaults to "./user_data/trials.csv".
        event_csv_path (str, optional): Relative path to event csv.
            Defaults to "./user_data/events.csv".
        skip_duplicates (bool, optional): See DataJoint `insert` function. Default True.
        verbose (bool, optional): Print number inserted (i.e., table length change).
            Defaults to True.
    """
    csvs = [
        recording_csv_path,
        recording_csv_path,
        block_csv_path,
        block_csv_path,
        trial_csv_path,
        trial_csv_path,
        trial_csv_path,
        trial_csv_path,
        event_csv_path,
        event_csv_path,
        event_csv_path,
    ]
    tables = [
        event.BehaviorRecording(),
        event.BehaviorRecording.File(),
        trial.Block(),
        trial.Block.Attribute(),
        trial.TrialType(),
        trial.Trial(),
        trial.Trial.Attribute(),
        trial.BlockTrial(),
        event.EventType(),
        event.Event(),
        trial.TrialEvent(),
    ]

    # Allow direct insert required because element-event has Imported that should be Manual
    ingest_csv_to_table(
        csvs,
        tables,
        skip_duplicates=skip_duplicates,
        verbose=verbose,
        allow_direct_insert=True,
    )


def ingest_alignment(
    alignment_csv_path: str = "./user_data/alignments.csv",
    skip_duplicates: bool = True,
    verbose: bool = True,
):
    """Ingest event alignment data from local CSVs

    Note: This is duplicated across wf-array-ephys and wf-calcium-imaging

    Args:
        alignment_csv_path (str, optional): Relative path to event alignment csv.
            Defaults to "./user_data/alignments.csv".
        skip_duplicates (bool, optional): See DataJoint `insert` function. Default True.
        verbose (bool, optional): Print number inserted (i.e., table length change).
            Defaults to True.
    """

    csvs = [alignment_csv_path]
    tables = [event.AlignmentEvent()]

    ingest_csv_to_table(csvs, tables, skip_duplicates=skip_duplicates, verbose=verbose)


if __name__ == "__main__":
    ingest_subjects()
    ingest_sessions()
    ingest_events()
    ingest_alignment()
