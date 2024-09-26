from __future__ import annotations

import numpy as np
import pandas as pd

"""
Geometry definition for Neuropixels probes
The definition here are all from Jennifer Colonell
See:
https://github.com/jenniferColonell/SGLXMetaToCoords/blob/main/SGLXMetaToCoords.py

A better approach is to pip install and use as a package
Unfortunately, the GitHub repo above is not yet packaged and pip installable

Better yet, full integration with ProbeInterface and the probes' geometry
 from Jennifer Colonell - this is in the making!

Latest update: 07-26-2023
"""

# many part numbers have the same geometry parameters ;
# define those sets in lists
# [nShank, shankWidth, shankPitch, even_xOff, odd_xOff, horizPitch, vertPitch, rowsPerShank, elecPerShank]
geom_param_names = [
    "nShank",
    "shankWidth",
    "shankPitch",
    "even_xOff",
    "odd_xOff",
    "horizPitch",
    "vertPitch",
    "rowsPerShank",
    "elecPerShank",
]

# offset and pitch values in um
np1_stag_70um = [1, 70, 0, 27, 11, 32, 20, 480, 960]
nhp_lin_70um = [1, 70, 0, 27, 27, 32, 20, 480, 960]
nhp_stag_125um_med = [1, 125, 0, 27, 11, 87, 20, 1368, 2496]
nhp_stag_125um_long = [1, 125, 0, 27, 11, 87, 20, 2208, 4416]
nhp_lin_125um_med = [1, 125, 0, 11, 11, 103, 20, 1368, 2496]
nhp_lin_125um_long = [1, 125, 0, 11, 11, 103, 20, 2208, 4416]
uhd_8col_1bank = [1, 70, 0, 14, 14, 6, 6, 48, 384]
uhd_8col_16bank = [1, 70, 0, 14, 14, 6, 6, 768, 6144]
np2_ss = [1, 70, 0, 27, 27, 32, 15, 640, 1280]
np2_4s = [4, 70, 250, 27, 27, 32, 15, 640, 1280]
NP1120 = [1, 70, 0, 6.75, 6.75, 4.5, 4.5, 192, 384]
NP1121 = [1, 70, 0, 6.25, 6.25, 3, 3, 384, 384]
NP1122 = [1, 70, 0, 6.75, 6.75, 4.5, 4.5, 24, 384]
NP1123 = [1, 70, 0, 10.25, 10.25, 3, 3, 32, 384]
NP1300 = [1, 70, 0, 11, 11, 48, 20, 480, 960]
NP1200 = [1, 70, 0, 27, 11, 32, 20, 64, 128]
NXT3000 = [1, 70, 0, 53, 53, 0, 15, 128, 128]

"""
Electrode coordinate system - from Bill Karsh
(https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/Metadata_30.md)

The X-origin is the left edge of the shank
The Y-origin is the center of the bottom-most elecrode row (closest to the tip) 
"""


M = dict(
    [
        ("3A", np1_stag_70um),
        ("PRB_1_4_0480_1", np1_stag_70um),
        ("PRB_1_4_0480_1_C", np1_stag_70um),
        ("NP1010", np1_stag_70um),
        ("NP1011", np1_stag_70um),
        ("NP1012", np1_stag_70um),
        ("NP1013", np1_stag_70um),
        ("NP1015", nhp_lin_70um),
        ("NP1016", nhp_lin_70um),
        ("NP1017", nhp_lin_70um),
        ("NP1020", nhp_stag_125um_med),
        ("NP1021", nhp_stag_125um_med),
        ("NP1030", nhp_stag_125um_long),
        ("NP1031", nhp_stag_125um_long),
        ("NP1022", nhp_lin_125um_med),
        ("NP1032", nhp_lin_125um_long),
        ("NP1100", uhd_8col_1bank),
        ("NP1110", uhd_8col_16bank),
        ("PRB2_1_4_0480_1", np2_ss),
        ("PRB2_1_2_0640_0", np2_ss),
        ("NP2000", np2_ss),
        ("NP2003", np2_ss),
        ("NP2004", np2_ss),
        ("PRB2_4_2_0640_0", np2_4s),
        ("PRB2_4_4_0480_1", np2_4s),
        ("NP2010", np2_4s),
        ("NP2013", np2_4s),
        ("NP2014", np2_4s),
        ("NP1120", NP1120),
        ("NP1121", NP1121),
        ("NP1122", NP1122),
        ("NP1123", NP1123),
        ("NP1300", NP1300),
        ("NP1200", NP1200),
        ("NXT3000", NXT3000),
    ]
)

# additional alias to maintain compatibility with previous naming in the pipeline
M["neuropixels 1.0 - 3A"] = M["3A"]
M["neuropixels 1.0 - 3B"] = M["NP1010"]
M["neuropixels 1.0"] = M["NP1010"]
M["neuropixels UHD"] = M["NP1100"]
M["neuropixels 2.0 - SS"] = M["NP2000"]
M["neuropixels 2.0 - MS"] = M["NP2010"]


def build_npx_probe(
    nShank: int,
    shankWidth: float,
    shankPitch: float,
    even_xOff: float,
    odd_xOff: float,
    horizPitch: float,
    vertPitch: float,
    rowsPerShank: int,
    elecPerShank: int,
    probe_type: str = "neuropixels",
):
    row_offset = np.tile([even_xOff, odd_xOff], int(rowsPerShank / 2))

    elec_pos_df = build_electrode_layouts(
        probe_type=probe_type,
        site_count_per_shank=elecPerShank,
        col_spacing=horizPitch,
        row_spacing=vertPitch,
        row_offset=row_offset,
        col_count_per_shank=int(elecPerShank / rowsPerShank),
        shank_count=nShank,
        shank_spacing=shankPitch,
        y_origin="bottom",
        as_dataframe=True,
    )

    return elec_pos_df


def to_probeinterface(electrodes_df, **kwargs):
    import probeinterface as pi

    probe_df = electrodes_df.copy()
    probe_df.rename(
        columns={
            "electrode": "contact_ids",
            "shank": "shank_ids",
            "x_coord": "x",
            "y_coord": "y",
        },
        inplace=True,
    )
    # Get the contact shapes. By default, it's set to circle with a radius of 10.
    contact_shapes = kwargs.get("contact_shapes", "circle")
    assert (
        contact_shapes in pi.probe._possible_contact_shapes
    ), f"contacts shape should be in {pi.probe._possible_contact_shapes}"

    probe_df["contact_shapes"] = contact_shapes
    if contact_shapes == "circle":
        probe_df["radius"] = kwargs.get("radius", 10)
    elif contact_shapes == "square":
        probe_df["width"] = kwargs.get("width", 10)
    elif contact_shapes == "rect":
        probe_df["width"] = kwargs.get("width")
        probe_df["height"] = kwargs.get("height")

    return pi.Probe.from_dataframe(probe_df)


def build_electrode_layouts(
    probe_type: str,
    site_count_per_shank: int,
    col_spacing: float = None,
    row_spacing: float = None,
    row_offset: list = None,
    col_count_per_shank: int = 1,
    shank_count: int = 1,
    shank_spacing: float = None,
    y_origin="bottom",
    as_dataframe=False,
) -> list[dict]:
    """Builds electrode layouts.

    Args:
        probe_type (str): probe type (e.g., "neuropixels 1.0 - 3A").
        site_count_per_shank (int): site count per shank.
        col_spacing (float): (um) horizontal spacing between sites. Defaults to None (single column).
        row_spacing (float): (um) vertical spacing between columns. Defaults to None (single row).
        row_offset (list): (um) per-row offset spacing. Defaults to None.
        col_count_per_shank (int): number of column per shank. Defaults to 1 (single column).
        shank_count (int): number of shank. Defaults to 1 (single shank).
        shank_spacing (float): (um) spacing between shanks. Defaults to None (single shank).
        y_origin (str): {"bottom", "top"}. y value decrements if "top". Defaults to "bottom".
        as_dataframe (bool): if True, returns as pandas DataFrame, otherwise as list of dict
    """
    row_count = int(site_count_per_shank / col_count_per_shank)
    x_coords = np.tile(
        np.arange(0, (col_spacing or 1) * col_count_per_shank, (col_spacing or 1)),
        row_count,
    )
    y_coords = np.repeat(np.arange(row_count) * (row_spacing or 1), col_count_per_shank)

    if row_offset is None:
        row_offset = np.zeros_like(x_coords)
    else:
        assert len(row_offset) == row_count
        row_offset = np.repeat(row_offset, col_count_per_shank)
    x_coords = x_coords + row_offset

    shank_cols = np.tile(range(col_count_per_shank), row_count)
    shank_rows = np.repeat(range(row_count), col_count_per_shank)

    electrode_layout = [
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

    return pd.DataFrame(electrode_layout) if as_dataframe else electrode_layout
