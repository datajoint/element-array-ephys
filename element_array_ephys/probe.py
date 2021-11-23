"""
Neuropixels Probes
"""

import datajoint as dj
import numpy as np

schema = dj.schema()


def activate(schema_name, *, create_schema=True, create_tables=True):
    """
    activate(schema_name, create_schema=True, create_tables=True)
        :param schema_name: schema name on the database server to activate the `probe` element
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)

    # Add neuropixels probes
    for probe_type in ('neuropixels 1.0 - 3A', 'neuropixels 1.0 - 3B', 'neuropixels UHD',
                       'neuropixels 2.0 - SS', 'neuropixels 2.0 - MS'):
        ProbeType.create_neuropixels_probe(probe_type)


@schema
class ProbeType(dj.Lookup):
    definition = """
    # Type of probe, with specific electrodes geometry defined
    probe_type: varchar(32)  # e.g. neuropixels_1.0
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        electrode: int       # electrode index, starts at 0
        ---
        shank: int           # shank index, starts at 0, advance left to right
        shank_col: int       # column index, starts at 0, advance left to right
        shank_row: int       # row index, starts at 0, advance tip to tail
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe, (0, 0) is the bottom left corner of the probe
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe, (0, 0) is the bottom left corner of the probe
        """

    @staticmethod
    def create_neuropixels_probe(probe_type='neuropixels 1.0 - 3A'):
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
            'neuropixels 1.0 - 3A': dict(site_count=960, col_spacing=32, row_spacing=20,
                                         white_spacing=16, col_count=2,
                                         shank_count=1, shank_spacing=0),
            'neuropixels 1.0 - 3B': dict(site_count=960, col_spacing=32, row_spacing=20,
                                         white_spacing=16, col_count=2,
                                         shank_count=1, shank_spacing=0),
            'neuropixels UHD': dict(site_count=384, col_spacing=6, row_spacing=6,
                                    white_spacing=0, col_count=8,
                                    shank_count=1, shank_spacing=0),
            'neuropixels 2.0 - SS': dict(site_count=1280, col_spacing=32, row_spacing=15,
                                         white_spacing=0, col_count=2,
                                         shank_count=1, shank_spacing=250),
            'neuropixels 2.0 - MS': dict(site_count=1280, col_spacing=32, row_spacing=15,
                                         white_spacing=0, col_count=2,
                                         shank_count=4, shank_spacing=250)
        }

        def build_electrodes(site_count, col_spacing, row_spacing,
                             white_spacing, col_count,
                             shank_count, shank_spacing):
            """
            :param site_count: site count per shank
            :param col_spacing: (um) horrizontal spacing between sites
            :param row_spacing: (um) vertical spacing between columns
            :param white_spacing: (um) offset spacing
            :param col_count: number of column per shank
            :param shank_count: number of shank
            :param shank_spacing: spacing between shanks
            :return:
            """
            row_count = int(site_count / col_count)
            x_coords = np.tile(np.arange(0, col_spacing * col_count, col_spacing), row_count)
            y_coords = np.repeat(np.arange(row_count) * row_spacing, col_count)

            if white_spacing:
                x_white_spaces = np.tile([white_spacing, white_spacing, 0, 0], int(row_count / 2))
                x_coords = x_coords + x_white_spaces

            shank_cols = np.tile(range(col_count), row_count)
            shank_rows = np.repeat(range(row_count), col_count)

            npx_electrodes = []
            for shank_no in range(shank_count):
                npx_electrodes.extend([{'electrode': (site_count * shank_no) + e_id,
                                        'shank': shank_no,
                                        'shank_col': c_id,
                                        'shank_row': r_id,
                                        'x_coord': x + (shank_no * shank_spacing),
                                        'y_coord': y}
                                       for e_id, (c_id, r_id, x, y) in enumerate(
                    zip(shank_cols, shank_rows, x_coords, y_coords))])

            return npx_electrodes

        electrodes = build_electrodes(**neuropixels_probes_config[probe_type])
        probe_type = {'probe_type': probe_type}
        with ProbeType.connection.transaction:
            ProbeType.insert1(probe_type, skip_duplicates=True)
            ProbeType.Electrode.insert([{**probe_type, **e} for e in electrodes],
                                       skip_duplicates=True)


@schema
class Probe(dj.Lookup):
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
    electrode_config_name: varchar(64)  # user friendly name
    """

    class Electrode(dj.Part):
        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
        """
