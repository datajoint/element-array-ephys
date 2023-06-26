import importlib
import inspect

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

schema = dj.schema()

_linking_module = None


def activate(
    schema_name, *, create_schema=True, create_tables=True, linking_module=None
):
    """Activate this schema.

    Args:
        schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (str): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.
    """
    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module), (
        "The argument 'dependency' must " + "be a module's name or a module"
    )

    global _linking_module
    _linking_module = linking_module

    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=linking_module.__dict__,
    )


@schema
class SpikesAlignmentCondition(dj.Manual):
    """Alignment activity table

    Attributes:
        ephys.CuratedClustering (foreign key): CuratedClustering foreign key
        event.AlignmentEvent (foreign key): AlignmentEvent foreign key
        trial_condition: varchar(128) # user-friendly name of condition
        condition_description ( varchar(1000), nullable): condition description
        bin_size (float, optional): Bin-size (in second) used to compute the PSTH
            Default 0.04
    """

    definition = """
    -> ephys.CuratedClustering
    -> event.AlignmentEvent
    trial_condition: varchar(128) # user-friendly name of condition
    ---
    condition_description='': varchar(1000)
    bin_size=0.04: float # bin-size (in second) used to compute the PSTH
    """

    class Trial(dj.Part):
        """Trials on which to compute event-aligned spikes and PSTH"""

        definition = """  # Trials on which to compute event-aligned spikes and PSTH
        -> master
        -> trial.Trial
        """


@schema
class SpikesAlignment(dj.Computed):
    """Spike alignment table pairing AlignedTrialSpikes and by-unit PSTH"""

    definition = """
    -> SpikesAlignmentCondition
    """

    class AlignedTrialSpikes(dj.Part):
        """Spike activity aligned to the event time within the designated window

        Attributes:
            SpikesAlignmentCondition.Trial (foreign key): Trial foreign key
            ephys.CuratedClustering.Unit (foreign key): Unit foreign key
            SpikesAlignmentCondition.Trial (foreign key): Trial foreign key
            aligned_spike_times (longblob): (s) spike times relative to alignment event
        """

        definition = """
        -> master
        -> ephys.CuratedClustering.Unit
        -> SpikesAlignmentCondition.Trial
        ---
        aligned_spike_times: longblob # (s) spike times relative to alignment event time
        """

    class UnitPSTH(dj.Part):
        """Event-aligned spike peristimulus time histogram (PSTH) by unit

        Attributes:
            SpikesAlignment (foreign key): SpikesAlignment foreign key
            ephys.CuratedClustering.Unit (foreign key): Unit foreign key
            psth (longblob): event-aligned spike peristimulus time histogram (PSTH)
            psth_edges (longblob): set of PSTH edges
        """

        definition = """
        -> master
        -> ephys.CuratedClustering.Unit
        ---
        psth: longblob  # event-aligned spike peristimulus time histogram (PSTH)
        psth_edges: longblob
        """

    def make(self, key: dict):
        """Populate SpikesAlignment, AlignedTrialSpikes and UnitPSTH

        Args:
            key (dict): Dict uniquely identifying one SpikesAlignmentCondition
        """
        unit_keys, unit_spike_times = (
            _linking_module.ephys.CuratedClustering.Unit & key
        ).fetch("KEY", "spike_times", order_by="unit")
        bin_size = (SpikesAlignmentCondition & key).fetch1("bin_size")

        trialized_event_times = (
            _linking_module.trial.get_trialized_alignment_event_times(
                key,
                _linking_module.trial.Trial & (SpikesAlignmentCondition.Trial & key),
            )
        )

        min_limit = (trialized_event_times.event - trialized_event_times.start).max()
        max_limit = (trialized_event_times.end - trialized_event_times.event).max()

        # Spike raster
        aligned_trial_spikes = []
        units_spike_raster = {
            u["unit"]: {**key, **u, "aligned_spikes": []} for u in unit_keys
        }
        for _, r in trialized_event_times.iterrows():
            if np.isnan(r.event):
                continue
            alignment_start_time = r.event - min_limit
            alignment_end_time = r.event + max_limit
            for unit_key, spikes in zip(unit_keys, unit_spike_times):
                aligned_spikes = (
                    spikes[
                        (alignment_start_time <= spikes) & (spikes < alignment_end_time)
                    ]
                    - r.event
                )
                aligned_trial_spikes.append(
                    {
                        **key,
                        **unit_key,
                        **r.trial_key,
                        "aligned_spike_times": aligned_spikes,
                    }
                )
                units_spike_raster[unit_key["unit"]]["aligned_spikes"].append(
                    aligned_spikes
                )

        # PSTH
        for unit_spike_raster in units_spike_raster.values():
            spikes = np.concatenate(unit_spike_raster["aligned_spikes"])

            psth, edges = np.histogram(
                spikes, bins=np.arange(-min_limit, max_limit, bin_size)
            )
            unit_spike_raster["psth"] = (
                psth / len(unit_spike_raster.pop("aligned_spikes")) / bin_size
            )
            unit_spike_raster["psth_edges"] = edges[1:]

        self.insert1(key)
        self.AlignedTrialSpikes.insert(aligned_trial_spikes)
        self.UnitPSTH.insert(list(units_spike_raster.values()))

    def plot(self, key: dict, unit: int, axs: tuple = None) -> Figure:
        """Plot event-aligned and trial-averaged spiking

        Args:
            key (dict): key of SpikesAlignmentCondition master table
            unit (int): ID of ephys.CuratedClustering.Unit table
            axs (tuple, optional): Definition of axes for plot.
                Default is plt.subplots(2, 1, figsize=(12, 8))

        Returns:
            fig (matplotlib.figure.Figure): Plot event-aligned and trial-averaged spikes
        """
        from .plotting import plot_psth

        fig = None
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        bin_size = (SpikesAlignmentCondition & key).fetch1("bin_size")
        trial_ids, aligned_spikes = (
            self.AlignedTrialSpikes & key & {"unit": unit}
        ).fetch("trial_id", "aligned_spike_times")
        psth, psth_edges = (self.UnitPSTH & key & {"unit": unit}).fetch1(
            "psth", "psth_edges"
        )

        xlim = psth_edges[0], psth_edges[-1]

        plot_psth._plot_spike_raster(
            aligned_spikes,
            trial_ids=trial_ids,
            ax=axs[0],
            title=f"{dict(**key, unit=unit)}",
            xlim=xlim,
        )
        plot_psth._plot_psth(psth, psth_edges, bin_size, ax=axs[1], title="", xlim=xlim)

        return fig
