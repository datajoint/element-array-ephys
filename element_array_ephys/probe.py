import datajoint as dj

from .readers import probe_geometry
from .readers.probe_geometry import build_electrode_layouts

log = dj.logger

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
        log.warning(
            "Class method `ProbeType.create_neuropixels_probe` is deprecated. Use `create_neuropixels_probe` instead.",
        )
        return create_neuropixels_probe(probe_type)


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


def create_neuropixels_probe_types():
    # Add neuropixels probes
    for probe_type in (
        "neuropixels 1.0 - 3A",
        "neuropixels 1.0 - 3B",
        "neuropixels UHD",
        "neuropixels 2.0 - SS",
        "neuropixels 2.0 - MS",
    ):
        if not (ProbeType & {"probe_type": probe_type}):
            create_neuropixels_probe(probe_type)


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
    Electrode numbering is 0-indexing
    """
    npx_probes_config = probe_geometry.M
    if probe_type not in npx_probes_config:
        raise ValueError(
            f"Probe type {probe_type} not found in probe_geometry configuration. Not a Neuropixels probe?"
        )

    probe_params = dict(
        zip(
            probe_geometry.geom_param_names,
            npx_probes_config[probe_type],
        )
    )
    electrode_layouts = probe_geometry.build_npx_probe(
        **{**probe_params, "probe_type": probe_type}
    )
    with ProbeType.connection.transaction:
        ProbeType.insert1({"probe_type": probe_type})
        ProbeType.Electrode.insert(electrode_layouts)
