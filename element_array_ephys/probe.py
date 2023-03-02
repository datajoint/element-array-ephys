from __future__ import annotations

import datajoint as dj
import numpy as np
from element_interface.utils import dict_to_uuid

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
        probe_type (foreign key, varchar (32) ): Name of the probe type.
    """

    definition = """
    # Type of probe, with specific electrodes geometry defined
    probe_type: varchar(32)  # e.g. neuropixels_1.0
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
    definition = """
    # The electrode configuration setting on a given probe
    electrode_config_hash: uuid
    ---
    -> ProbeType
    electrode_config_name: varchar(4000)  # user friendly name
    """

    class Electrode(dj.Part):
        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
        ---
        channel  : varchar(16) # channel name fetched from raw data (e.g., A-001)
        """


def build_electrode_layouts(
    probe_type: str,
    site_count_per_shank: int,
    col_spacing: float = None,
    row_spacing: float = None,
    white_spacing: float = None,
    col_count_per_shank: int = 1,
    shank_count: int = 1,
    shank_spacing: float = None,
    y_origin="bottom",
) -> list[dict]:
    """Builds electrode layouts.

    Args:
        probe_type (str): probe type (e.g., "neuropixels 1.0 - 3A").
        site_count_per_shank (int): site count per shank.
        col_spacing (float): (um) horizontal spacing between sites. Defaults to None (single column).
        row_spacing (float): (um) vertical spacing between columns. Defaults to None (single row).
        white_spacing (float): (um) offset spacing. Defaults to None.
        col_count_per_shank (int): number of column per shank. Defaults to 1 (single column).
        shank_count (int): number of shank. Defaults to 1 (single shank).
        shank_spacing (float): (um) spacing between shanks. Defaults to None (single shank).
        y_origin (str): {"bottom", "top"}. y value decrements if "top". Defaults to "bottom".
    """
    row_count = int(site_count_per_shank / col_count_per_shank)
    x_coords = np.tile(
        np.arange(0, (col_spacing or 1) * col_count_per_shank, (col_spacing or 1)),
        row_count,
    )
    y_coords = np.repeat(np.arange(row_count) * (row_spacing or 1), col_count_per_shank)

    if white_spacing:
        x_white_spaces = np.tile(
            [white_spacing, white_spacing, 0, 0], int(row_count / 2)
        )
        x_coords = x_coords + x_white_spaces

    shank_cols = np.tile(range(col_count_per_shank), row_count)
    shank_rows = np.repeat(range(row_count), col_count_per_shank)

    return [
        {
            "probe_type": probe_type,
            "electrode": (site_count_per_shank * shank_no) + e_id,
            "shank": shank_no,
            "shank_col": c_id,
            "shank_row": r_id,
            "x_coord": x + (shank_no * (shank_spacing or 1)),
            "y_coord": {"top": -y, "bottom": y}[y_origin],
        }
        for shank_no in range(shank_count)
        for e_id, (c_id, r_id, x, y) in enumerate(
            zip(shank_cols, shank_rows, x_coords, y_coords)
        )
    ]
