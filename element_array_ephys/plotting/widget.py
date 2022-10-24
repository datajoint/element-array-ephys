import ipywidgets as widgets
from .. import ephys_report
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib
import typing as T
from IPython.display import display, clear_output
from .. import ephys_no_curation as ephys
from skimage import io


# Build selection widgets
probe_dropdown_wg = widgets.Dropdown(
    options=ephys.CuratedClustering & ephys_report.ProbeLevelReport.fetch("KEY"),
    description="Select Probe Insertion : ",
    disabled=False,
    layout=widgets.Layout(
        width="80%",
    ),
    style={"description_width": "150px"},
)

shank_dropdown_wg = widgets.Dropdown(
    options=(ephys_report.ProbeLevelReport & probe_dropdown_wg.value).fetch("shank"),
    description="Select Shank : ",
    disabled=False,
    layout=widgets.Layout(
        width="15%",
    ),
    style={"description_width": "100px"},
)

shank_toggle_wg = widgets.ToggleButtons(
    options=(ephys_report.ProbeLevelReport & probe_dropdown_wg.value).fetch("shank"),
    description="Select Shank : ",
    layout=widgets.Layout(width="auto"),
    style={"button_width": "50px"},
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


def probe_widget():
    def plot_probe_figure(probe_key, shank):

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
        return go.FigureWidget(probe_fig)

    def probe_dropdown_evt(change):
        probe_key = change.new
        shank_dropdown_wg.options = (
            ephys_report.ProbeLevelReport & probe_key.value
        ).fetch("shank")
        unit_dropdown_wg.options = (
            (
                (ephys_report.UnitLevelReport & probe_key.value)
                & "cluster_quality_label='good'"
            ).fetch("unit"),
        )
        clear_output()
        display(
            widgets.VBox(
                [
                    probe_dropdown_wg,
                    shank_dropdown_wg,
                    plot_probe_figure(probe_key, shank_dropdown_wg.value),
                ]
            )
        )

    probe_dropdown_wg.observe(probe_dropdown_evt, "value")

    return widgets.VBox(
        [
            probe_dropdown_wg,
            shank_dropdown_wg,
            plot_probe_figure(probe_dropdown_wg.value, shank_dropdown_wg.value),
        ]
    )


def unit_widget():
    def plot_unit_figure(unit):
        # Build a unit widgets
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
        return unit_fig_wg

    def unit_dropdown_evt(change):
        unit = change.new
        clear_output()
        display(unit_dropdown_wg, plot_unit_figure(unit))

    unit_dropdown_wg.observe(unit_dropdown_evt, "value")
    return widgets.VBox([unit_dropdown_wg, plot_unit_figure(unit_dropdown_wg.value)])
