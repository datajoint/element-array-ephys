from __future__ import annotations

import datajoint as dj

from .readers import probe_geometry

schema = dj.schema()


def activate(
    schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
):
    """Activates the `probe` schemas.

    Args:
        schema_name (str): A string containing the name of the probe schema.
        create_schema (bool): If True, schema will be created in the database.
        create_tables (bool): If True, tables related to the schema will be created in the database.
    """
    schema.activate(
        schema_name, create_schema=create_schema, create_tables=create_tables
    )


@schema
class ProbeType(dj.Lookup):
    """Type of probe.

    Attributes:
        probe_type (foreign key, varchar (64) ): Name of the probe type.
        probe_full_name ( varchar(100) ): full, non-abbreviated name of the probe.
    """

    definition = """
    # Type of probe, with specific electrodes geometry defined
    probe_type           : varchar(64) # e.g. A1x32-6mm-100-177-H32_21mm
    ---
    probe_full_name=null : varchar(100) # full, non-abbreviated name of the probe 
    """

    class Electrode(dj.Part):
        """Electrode information for a given probe.

        Attributes:
            ProbeType (foreign key): ProbeType primary key.
            electrode (foreign key, int): Electrode index, starting at 0.
            shank (int): shank index, starting at 0.
            shank_col (int): column index, starting at 0.
            shank_row (int): row index, starting at 0.
            x_coord (float): x-coordinate of the electrode within the probe in micrometers.
            y_coord (float): y-coordinate of the electrode within the probe in micrometers.
        """

        definition = """
        -> master
        electrode: int       # electrode index, starts at 0
        ---
        shank: int           # shank index, starts at 0, advance left to right
        shank_col: int       # column index, starts at 0, advance left to right
        shank_row: int       # row index, starts at 0.
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe.
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe.
        """


@schema
class Probe(dj.Lookup):
    """Represent a physical probe with unique ID

    Attributes:
        probe (foreign key, varchar(32) ): Unique ID for this model of the probe.
        ProbeType (dict): ProbeType entry.
        probe_comment ( varchar(1000) ): Comment about this model of probe.
    """

    definition = """
    # Represent a physical probe with unique identification
    probe: varchar(32)  # unique identifier for this model of probe (e.g. serial number)
    ---
    -> ProbeType
    probe_comment='' :  varchar(1000)
    """


@schema
class ElectrodeConfig(dj.Lookup):
    """Electrode configuration setting on a probe.

    Attributes:
        electrode_config_hash (foreign key, uuid): unique index for electrode configuration.
        ProbeType (dict): ProbeType entry.
        electrode_config_name ( varchar(4000) ): User-friendly name for this electrode configuration.
    """

    definition = """
    # The electrode configuration setting on a given probe
    electrode_config_hash: uuid
    ---
    -> ProbeType
    electrode_config_name: varchar(4000)  # user friendly name
    """

    class Electrode(dj.Part):
        """Electrode included in the recording.

        Attributes:
            ElectrodeConfig (foreign key): ElectrodeConfig primary key.
            ProbeType.Electrode (foreign key): ProbeType.Electrode primary key.
            channel ( varchar(16) ): channel name fetched from raw data (e.g., A-001).
        """

        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
        ---
        channel_idx: smallint unsigned  # channel index (index of the raw data)
        """
