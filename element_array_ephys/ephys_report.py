import pathlib
import datetime
import datajoint as dj
import typing as T

from element_interface.utils import insert1_skip_full_duplicates

schema = dj.schema()

ephys = None


def activate(schema_name, ephys_schema_name, *, create_schema=True, create_tables=True):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `ephys_report` schema
        :param ephys_schema_name: schema name of the activated ephys element for which this ephys_report schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    (The "activation" of this ephys_report module should be evoked by one of the ephys modules only)
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
    definition = """
    -> ephys.CuratedClustering
    shank         : tinyint unsigned
    ---
    drift_map_plot: attach
    """

    def make(self, key):

        from . import probe
        from .plotting.probe_level import plot_driftmap

        save_dir = _make_save_dir()

        units = ephys.CuratedClustering.Unit & key & "cluster_quality_label='good'"

        shanks = set((probe.ProbeType.Electrode & units).fetch("shank"))

        for shank_no in shanks:

            table = (
                units
                * ephys.ProbeInsertion.proj()
                * probe.ProbeType.Electrode.proj("shank")
                & {"shank": shank_no}
            )

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
    definition = """
    -> ephys.CuratedClustering.Unit
    ---
    cluster_quality_label   : varchar(100) 
    waveform_plotly         : longblob  
    autocorrelogram_plotly  : longblob
    depth_waveform_plotly   : longblob
    """

    def make(self, key):

        from .plotting.unit_level import (
            plot_waveform,
            plot_auto_correlogram,
            plot_depth_waveforms,
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
    definition = """
    cutoffs_id                    : smallint
    ---
    amplitude_cutoff_maximum=null : float # Defualt null, no cutoff applied
    presence_ratio_minimum=null   : float # Defualt null, no cutoff applied
    isi_violations_maximum=null   : float # Defualt null, no cutoff applied
    """

    contents = [(0, None, None, None), (1, 0.1, 0.9, 0.5)]

    def insert_new_cutoffs(
        cls,
        cutoffs_id: int,
        amplitude_cutoff_maximum: float,
        presence_ratio_minimum: float,
        isi_violations_maximum: float,
    ):
        insert1_skip_full_duplicates(  # depends on element-interface/pull/43
            cls,
            dict(
                cutoffs_id=cutoffs_id,
                amplitude_cutoff_maximum=amplitude_cutoff_maximum,
                presence_ratio_minimum=presence_ratio_minimum,
                isi_violations_maximum=isi_violations_maximum,
            ),
        )


@schema
class QualityMetricSet(dj.Manual):
    definition = """
    -> ephys.QualityMetrics
    -> QualityMetricCutoffs
    """


@schema
class QualityMetricReport(dj.Computed):
    definition = """
    -> QualityMetricSet
    """

    class Cluster(dj.Part):
        definition = """
        -> master
        ---
        firing_rate_plot    : longblob
        presence_ratio_plot : longblob
        amp_cutoff_plot     : longblob
        isi_violation_plot  : longblob
        snr_plot            : longblob
        iso_dist_plot       : longblob
        d_prime_plot        : longblob
        nn_hit_plot         : longblob
        """

    def make(self, key):
        from .plotting.qc import QualityMetricFigs

        cutoffs = (QualityMetricCutoffs & key).fetch1()
        qc_key = ephys.QualityMetrics & key
        qc_figs = QualityMetricFigs(qc_key, **cutoffs)

        # ADD SAVEFIG?

        self.insert1(key)
        self.Cluster.insert1(
            dict(
                **key,
                firing_rate_plot=qc_figs.firing_rate_plot().to_plotly_json(),
                presence_ratio_plot=qc_figs.presence_ratio_plot().to_plotly_json(),
                amp_cutoff_plot=qc_figs.amp_cutoff_plot().to_plotly_json(),
                isi_violation_plot=qc_figs.isi_violation_plot().to_plotly_json(),
                snr_plot=qc_figs.snr_plot().to_plotly_json(),
                iso_dist_plot=qc_figs.iso_dist_plot().to_plotly_json(),
                d_prime_plot=qc_figs.d_prime_plot().to_plotly_json(),
                nn_hit_plot=qc_figs.nn_hit_plot().to_plotly_json(),
            )
        )


def _make_save_dir(root_dir: pathlib.Path = None) -> pathlib.Path:
    if root_dir is None:
        root_dir = pathlib.Path().absolute()
    save_dir = root_dir / "ephys_figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _save_figs(
    figs, fig_names, save_dir, fig_prefix, extension=".png"
) -> T.Dict[str, pathlib.Path]:
    fig_dict = {}
    for fig, fig_name in zip(figs, fig_names):
        fig_filepath = save_dir / (fig_prefix + "_" + fig_name + extension)
        fig.tight_layout()
        fig.savefig(fig_filepath)
        fig_dict[fig_name] = fig_filepath.as_posix()

    return fig_dict
