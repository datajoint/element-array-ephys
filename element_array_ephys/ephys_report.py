from __future__ import annotations

import datetime
import pathlib
from uuid import UUID

import datajoint as dj
from element_interface.utils import dict_to_uuid

from . import probe

schema = dj.schema()

ephys = None


def activate(schema_name, ephys_schema_name, *, create_schema=True, create_tables=True):
    """Activate the current schema.

    Args:
        schema_name (str): schema name on the database server to activate the `ephys_report` schema.
        ephys_schema_name (str): schema name of the activated ephys element for which
                this ephys_report schema will be downstream from.
        create_schema (bool, optional): If True (default), create schema in the database if it does not yet exist.
        create_tables (bool, optional): If True (default), create tables in the database if they do not yet exist.
    """

    global ephys
    ephys = dj.create_virtual_module("ephys", ephys_schema_name)
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=ephys.__dict__,
    )


@schema
class ProbeLevelReport(dj.Computed):
    """Table for storing probe level figures.

    Attributes:
        ephys.CuratedClustering (foreign key): ephys.CuratedClustering primary key.
        shank (tinyint unsigned): Shank of the probe.
        drift_map_plot (attach): Figure object for drift map.
    """

    definition = """
    -> ephys.CuratedClustering
    shank         : tinyint unsigned
    ---
    drift_map_plot: attach
    """

    def make(self, key):
        from .plotting.probe_level import plot_driftmap

        save_dir = _make_save_dir()

        units = ephys.CuratedClustering.Unit & key & "cluster_quality_label='good'"

        shanks = set((probe.ProbeType.Electrode & units).fetch("shank"))

        for shank_no in shanks:
            table = units * ephys.ProbeInsertion * probe.ProbeType.Electrode & {
                "shank": shank_no
            }

            spike_times, spike_depths = table.fetch(
                "spike_times", "spike_depths", order_by="unit"
            )

            # Get the figure
            fig = plot_driftmap(spike_times, spike_depths, colormap="gist_heat_r")
            fig_prefix = (
                "-".join(
                    [
                        v.strftime("%Y%m%d%H%M%S")
                        if isinstance(v, datetime.datetime)
                        else str(v)
                        for v in key.values()
                    ]
                )
                + f"-{shank_no}"
            )

            # Save fig and insert
            fig_dict = _save_figs(
                figs=(fig,),
                fig_names=("drift_map_plot",),
                save_dir=save_dir,
                fig_prefix=fig_prefix,
                extension=".png",
            )

            self.insert1({**key, **fig_dict, "shank": shank_no})


@schema
class UnitLevelReport(dj.Computed):
    """Table for storing unit level figures.

    Attributes:
        ephys.CuratedClustering.Unit (foreign key): ephys.CuratedClustering.Unit primary key.
        ephys.ClusterQualityLabel (foreign key): ephys.ClusterQualityLabel primary key.
        waveform_plotly (longblob): Figure object for unit waveform.
        autocorrelogram_plotly (longblob): Figure object for an autocorrelogram.
        depth_waveform_plotly (longblob): Figure object for depth waveforms.
    """

    definition = """
    -> ephys.CuratedClustering.Unit
    ---
    -> ephys.ClusterQualityLabel
    waveform_plotly                 : longblob
    autocorrelogram_plotly          : longblob
    depth_waveform_plotly           : longblob
    """

    def make(self, key):
        from .plotting.unit_level import (
            plot_auto_correlogram,
            plot_depth_waveforms,
            plot_waveform,
        )

        sampling_rate = (ephys.EphysRecording & key).fetch1(
            "sampling_rate"
        ) / 1e3  # in kHz

        peak_electrode_waveform, spike_times, cluster_quality_label = (
            (ephys.CuratedClustering.Unit & key) * ephys.WaveformSet.PeakWaveform
        ).fetch1("peak_electrode_waveform", "spike_times", "cluster_quality_label")

        # Get the figure
        waveform_fig = plot_waveform(
            waveform=peak_electrode_waveform, sampling_rate=sampling_rate
        )

        correlogram_fig = plot_auto_correlogram(
            spike_times=spike_times, bin_size=0.001, window_size=1
        )

        depth_waveform_fig = plot_depth_waveforms(ephys, unit_key=key, y_range=60)

        self.insert1(
            {
                **key,
                "cluster_quality_label": cluster_quality_label,
                "waveform_plotly": waveform_fig.to_plotly_json(),
                "autocorrelogram_plotly": correlogram_fig.to_plotly_json(),
                "depth_waveform_plotly": depth_waveform_fig.to_plotly_json(),
            }
        )


@schema
class QualityMetricCutoffs(dj.Lookup):
    """Cut-off values for unit quality metrics.

    Attributes:
        cutoffs_id (smallint): Unique ID for the cut-off values.
        amplitude_cutoff_maximum (float): Optional. Amplitude cut-off.
        presence_ratio_minimum (float): Optional. Presence ratio cut-off.
        isi_violations_maximum (float): Optional. ISI violation ratio cut-off.
        cutoffs_hash (uuid): uuid for the cut-off values.
    """

    definition = """
    cutoffs_id                    : smallint
    ---
    amplitude_cutoff_maximum=null : float # Defaults to null, no cutoff applied
    presence_ratio_minimum=null   : float # Defaults to null, no cutoff applied
    isi_violations_maximum=null   : float # Defaults to null, no cutoff applied
    cutoffs_hash: uuid
    unique index (cutoffs_hash)
    """

    contents = [
        (0, None, None, None, UUID("5d835de1-e1af-1871-d81f-d12a9702ff5f")),
        (1, 0.1, 0.9, 0.5, UUID("f74ccd77-0b3a-2bf8-0bfd-ec9713b5dca8")),
    ]

    @classmethod
    def insert_new_cutoffs(
        cls,
        cutoffs_id: int = None,
        amplitude_cutoff_maximum: float = None,
        presence_ratio_minimum: float = None,
        isi_violations_maximum: float = None,
    ):
        if cutoffs_id is None:
            cutoffs_id = (dj.U().aggr(cls, n="max(cutoffs_id)").fetch1("n") or 0) + 1

        param_dict = {
            "amplitude_cutoff_maximum": amplitude_cutoff_maximum,
            "presence_ratio_minimum": presence_ratio_minimum,
            "isi_violations_maximum": isi_violations_maximum,
        }
        param_hash = dict_to_uuid(param_dict)
        param_query = cls & {"cutoffs_hash": param_hash}

        if param_query:  # If the specified cutoff set already exists
            existing_paramset_idx = param_query.fetch1("cutoffs_id")
            if (
                existing_paramset_idx == cutoffs_id
            ):  # If the existing set has the same id: job done
                return
            # If not same name: human err, adding the same set with different name
            else:
                raise dj.DataJointError(
                    f"The specified param-set already exists"
                    f" - with paramset_idx: {existing_paramset_idx}"
                )
        else:
            if {"cutoffs_id": cutoffs_id} in cls.proj():
                raise dj.DataJointError(
                    f"The specified cuttoffs_id {cutoffs_id} already exists,"
                    f" please pick a different one."
                )
            cls.insert1(
                {"cutoffs_id": cutoffs_id, **param_dict, "cutoffs_hash": param_hash}
            )


@schema
class QualityMetricSet(dj.Manual):
    """Set of quality metric values for clusters and its cut-offs.

    Attributes:
        ephys.QualityMetrics (foreign key): ephys.QualityMetrics primary key.
        QualityMetricCutoffs (foreign key): QualityMetricCutoffs primary key.
    """

    definition = """
    -> ephys.QualityMetrics
    -> QualityMetricCutoffs
    """


@schema
class QualityMetricReport(dj.Computed):
    """Table for storing quality metric figures.

    Attributes:
        QualityMetricSet (foreign key): QualityMetricSet primary key.
        plot_grid (longblob): Plotly figure object.
    """

    definition = """
    -> QualityMetricSet
    ---
    plot_grid : longblob
    """

    def make(self, key):
        from .plotting.qc import QualityMetricFigs

        cutoffs = (QualityMetricCutoffs & key).fetch1()
        qc_key = ephys.QualityMetrics & key

        self.insert1(
            key.update(
                dict(plot_grid=QualityMetricFigs(qc_key, **cutoffs).get_grid().to_json)
            )
        )


def _make_save_dir(root_dir: pathlib.Path = None) -> pathlib.Path:
    if root_dir is None:
        root_dir = pathlib.Path().absolute()
    save_dir = root_dir / "temp_ephys_figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _save_figs(
    figs, fig_names, save_dir, fig_prefix, extension=".png"
) -> dict[str, pathlib.Path]:
    fig_dict = {}
    for fig, fig_name in zip(figs, fig_names):
        fig_filepath = save_dir / (fig_prefix + "_" + fig_name + extension)
        fig.tight_layout()
        fig.savefig(fig_filepath)
        fig_dict[fig_name] = fig_filepath.as_posix()

    return fig_dict
