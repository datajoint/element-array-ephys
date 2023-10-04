import pathlib
import types

import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display
from ipywidgets import widgets
from skimage import io

from .. import ephys_report


def main(ephys: types.ModuleType) -> widgets:
    # Build dropdown widgets
    probe_dropdown_wg = widgets.Dropdown(
        options=ephys.CuratedClustering & ephys_report.ProbeLevelReport,
        description="Select Probe Insertion : ",
        disabled=False,
        layout=widgets.Layout(
            width="80%",
        ),
        style={"description_width": "150px"},
    )

    shank_dropdown_wg = widgets.Dropdown(
        options=(ephys_report.ProbeLevelReport & probe_dropdown_wg.value).fetch(
            "shank"
        ),
        description="Select Shank : ",
        disabled=False,
        layout=widgets.Layout(
            width="15%",
        ),
        style={"description_width": "100px"},
    )

    unit_dropdown_wg = widgets.Dropdown(
        options=(
            (ephys_report.UnitLevelReport & probe_dropdown_wg.value)
            & "cluster_quality_label='good'"
        ).fetch("unit"),
        description="Select Units : ",
        disabled=False,
        layout=widgets.Layout(
            width="15%",
        ),
        style={"description_width": "100px"},
    )

    def probe_dropdown_evt(change):
        """Change in probe dropdown option triggers this function"""

        probe_key = change.new

        shank_dropdown_wg.options = (
            ephys_report.ProbeLevelReport & probe_key.value
        ).fetch("shank")

        unit_dropdown_wg.options = (
            (
                ephys_report.UnitLevelReport
                & probe_key.value
                & "cluster_quality_label='good'"
            ).fetch("unit"),
        )

    def plot_probe_widget(probe_key, shank):
        fig_name = (
            ephys_report.ProbeLevelReport & probe_key & f"shank={shank}"
        ).fetch1("drift_map_plot")

        # Constants
        img_width = 2000
        img_height = 1000
        scale_factor = 0.5

        img = io.imread(fig_name)
        probe_fig = px.imshow(img)

        # Configure other layout
        probe_fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 50, "r": 0, "t": 0, "b": 0},
            hovermode=False,
            xaxis_visible=False,
            yaxis_visible=False,
        )
        pathlib.Path(fig_name).unlink()
        display(go.FigureWidget(probe_fig))

    def plot_unit_widget(unit):
        waveform_fig, autocorrelogram_fig, depth_waveform_fig = (
            ephys_report.UnitLevelReport & probe_dropdown_wg.value & f"unit={unit}"
        ).fetch1("waveform_plotly", "autocorrelogram_plotly", "depth_waveform_plotly")
        waveform_fig = go.FigureWidget(waveform_fig).update_layout(
            width=300, height=300
        )
        autocorrelogram_fig = go.FigureWidget(autocorrelogram_fig).update_layout(
            width=300, height=300
        )
        depth_waveform_fig = go.FigureWidget(depth_waveform_fig)
        depth_waveform_fig.update_layout(
            width=300,
            height=600,
            autosize=False,
            margin={"l": 0, "r": 0, "t": 100, "b": 100},
        )

        unit_fig_wg = widgets.HBox(
            [widgets.VBox([waveform_fig, autocorrelogram_fig]), depth_waveform_fig],
            layout=widgets.Layout(margin="0 0 0 100px"),
        )
        display(unit_fig_wg)

    probe_dropdown_wg.observe(probe_dropdown_evt, "value")

    probe_widget = widgets.interactive(
        plot_probe_widget, probe_key=probe_dropdown_wg, shank=shank_dropdown_wg
    )

    unit_widget = widgets.interactive(plot_unit_widget, unit=unit_dropdown_wg)

    return widgets.VBox([probe_widget, unit_widget])
