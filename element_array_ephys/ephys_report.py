import json
import pathlib
import datetime
import datajoint as dj
import importlib
import inspect
import typing as T
from .plotting.unit_level import plot_waveform, plot_correlogram, plot_depth_waveforms
from .plotting.probe_level import plot_driftmap

schema = dj.schema()

ephys = None


def _activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `probe` element
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param activated_ephys: ephys module with the schema already activated
    (The "activation" of this ephys_report module should be evoked by one of the ephys modules only)
    """
    global ephys
    ephys = activated_ephys

    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)


@schema
class ProbeLevelReport(dj.Computed):
    definition = """
    -> ephys.CuratedClustering
    ---
    drift_map_plot: attach
    """

    def make(self, key):
        spike_times, spike_depths = (
            ephys.CuratedClustering.Unit & key & "cluster_quality_label='good'"
        ).fetch("spike_times", "spike_depths", order_by="unit")

        # Get the figure
        fig = plot_driftmap(spike_times, spike_depths, colormap="gist_heat_r")
        fig_prefix = "-".join(
            [
                v.strftime("%Y%m%d%H%M%S")
                if isinstance(v, datetime.datetime)
                else str(v)
                for v in key.values()
            ]
        )

        # Save fig and insert
        save_dir = _make_save_dir()
        fig_dict = _save_figs(
            figs=(fig,),
            fig_names=("drift_map_plot",),
            save_dir=save_dir,
            fig_prefix=fig_prefix,
            extension=".png",
        )

        self.insert1({**key, **fig_dict})


@schema
class UnitLevelReport(dj.Computed):
    definition = """
    -> ephys.CuratedClustering.Unit
    ---
    cluster_quality_label   : varchar(100) 
    waveform_plotly         : longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    autocorrelogram_plotly  : longblob
    depth_waveform_plotly   : longblob
    """

    def make(self, key):
        sampling_rate = (ephys.EphysRecording & key).fetch1(
            "sampling_rate"
        ) / 1e3  # in kHz

        peak_electrode_waveform, spike_times, cluster_quality_label = (
            (ephys.CuratedClustering.Unit & key)
            * ephys.WaveformSet.PeakWaveform
        ).fetch1("peak_electrode_waveform", "spike_times", "cluster_quality_label")

        # Get the figure
        fig = plot_waveform(
            waveform=peak_electrode_waveform, sampling_rate=sampling_rate
        )
        fig_waveform = json.loads(fig.to_json())

        fig = plot_correlogram(spike_times=spike_times, bin_size=0.001, window_size=1)
        fig_correlogram = json.loads(fig.to_json())

        fig = plot_depth_waveforms(ephys, unit_key=key, y_range=50)
        fig_depth_waveform = json.loads(fig.to_json())

        self.insert1(
            {
                **key,
                "cluster_quality_label": cluster_quality_label,
                "waveform_plotly": fig_waveform,
                "autocorrelogram_plotly": fig_correlogram,
                "depth_waveform_plotly": fig_depth_waveform,
            }
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
