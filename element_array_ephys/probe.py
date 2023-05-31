"""
Neuropixels Probes
"""
from __future__ import annotations

import datajoint as dj
import numpy as np

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

    Dependencies:
    Upstream tables:
        Session: A parent table to ProbeInsertion.

    Functions:
    """
    schema.activate(
        schema_name, create_schema=create_schema, create_tables=create_tables
    )

    # # Add neuropixels probes
    # for probe_type in (
    #     "neuropixels 1.0 - 3A",
    #     "neuropixels 1.0 - 3B",
    #     "neuropixels UHD",
    #     "neuropixels 2.0 - SS",
    #     "neuropixels 2.0 - MS",
    # ):
    #     if not (ProbeType & {"probe_type": probe_type}):
    #         try:
    #             ProbeType.create_neuropixels_probe(probe_type)
    #         except dj.errors.DataJointError as e:
    #             print(f"Unable to create probe-type: {probe_type}\n{str(e)}")


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

    @staticmethod
    def create_neuropixels_probe(probe_type: str = "neuropixels 1.0 - 3A"):
        """
        Create `ProbeType` and `Electrode` for neuropixels probes:
        + neuropixels 1.0 - 3A
        + neuropixels 1.0 - 3B
        + neuropixels UHD
        + neuropixels 2.0 - SS
        + neuropixels 2.0 - MS

        For electrode location, the (0, 0) is the
         bottom left corner of the probe (ignore the tip portion)
        Electrode numbering is 1-indexing
        """

        neuropixels_probes_config = {
            "neuropixels 1.0 - 3A": dict(
                site_count_per_shank=960,
                col_spacing=32,
                row_spacing=20,
                white_spacing=16,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=0,
            ),
            "neuropixels 1.0 - 3B": dict(
                site_count_per_shank=960,
                col_spacing=32,
                row_spacing=20,
                white_spacing=16,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=0,
            ),
            "neuropixels UHD": dict(
                site_count_per_shank=384,
                col_spacing=6,
                row_spacing=6,
                white_spacing=0,
                col_count_per_shank=8,
                shank_count=1,
                shank_spacing=0,
            ),
            "neuropixels 2.0 - SS": dict(
                site_count_per_shank=1280,
                col_spacing=32,
                row_spacing=15,
                white_spacing=0,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=250,
            ),
            "neuropixels 2.0 - MS": dict(
                site_count_per_shank=1280,
                col_spacing=32,
                row_spacing=15,
                white_spacing=0,
                col_count_per_shank=2,
                shank_count=4,
                shank_spacing=250,
            ),
        }

        probe_type = {"probe_type": probe_type}
        electrode_layouts = build_electrode_layouts(
            **{**neuropixels_probes_config[probe_type["probe_type"]], **probe_type}
        )
        with ProbeType.connection.transaction:
            ProbeType.insert1(probe_type, skip_duplicates=True)
            ProbeType.Electrode.insert(electrode_layouts, skip_duplicates=True)


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
        """

        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
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
    start_index: int = 0,
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
        start_index (int): starting index for electrode and shank indexes. Defaults to 0.

    Returns:
        list[dict]: List of dictionaries representing electrode layouts.
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
            "electrode": (site_count_per_shank * shank_no) + e_id + start_index,
            "shank": shank_no + start_index,
            "shank_col": c_id + start_index,
            "shank_row": r_id + start_index,
            "x_coord": x + (shank_no * (shank_spacing or 1)),
            "y_coord": {"top": -y, "bottom": y}[y_origin],
        }
        for shank_no in range(shank_count)
        for e_id, (c_id, r_id, x, y) in enumerate(
            zip(shank_cols, shank_rows, x_coords, y_coords)
        )]