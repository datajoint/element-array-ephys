import pathlib
from skimage import io
from modulefinder import Module
import ipywidgets as widgets
from ipywidgets import widgets as wg
from IPython.display import display
from plotly.io import from_json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px

from .. import ephys_report


def main(ephys: Module) -> widgets:

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


def qc_widget(ephys: Module) -> widgets:
    from .qc import QualityMetricFigs

    title_button = wg.Button(
        description="Ephys Quality Control Metrics",
        button_style="info",
        layout=wg.Layout(
            height="auto", width="auto", grid_area="title_button", border="solid"
        ),
        style=wg.ButtonStyle(button_color="#00a0df"),
        disabled=True,
    )

    cluster_dropdown = wg.Dropdown(
        options=ephys.QualityMetrics.fetch("KEY"),
        description="Clusters:",
        description_tooltip='Press "Load" to visualize the clusters identified.',
        disabled=False,
        layout=wg.Layout(
            width="95%",
            display="flex",
            flex_flow="row",
            justify_content="space-between",
            grid_area="cluster_dropdown",
        ),
        style={"description_width": "80px"},
    )

    cutoff_dropdown = wg.Dropdown(
        options=ephys_report.QualityMetricCutoffs.fetch("KEY"),
        description="Cutoffs:",
        description_tooltip='Press "Load" to visualize the clusters identified.',
        disabled=False,
        layout=wg.Layout(
            width="95%",
            display="flex",
            flex_flow="row",
            justify_content="space-between",
            grid_area="cutoff_dropdown",
        ),
        style={"description_width": "80px"},
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.01,
        vertical_spacing=0,
        column_titles=["Firing", "Title2"],
    )
    fwg = go.FigureWidget(fig)

    # figure_output = wg.VBox(
    #     [QualityMetricFigs.empty_fig()],
    #     layout=wg.Layout(width="95%", grid_area="figure_output"),
    # )
    # figure_output.add_class("box_style")

    load_button = wg.Button(
        description="Load",
        tooltip="Load figures.",
        layout=wg.Layout(width="auto", grid_area="load_button"),
    )

    def response(change, usedb=False):  # TODO: Accept cutoff vals?
        global firing_rate_plot
        if usedb:
            if cluster_dropdown.value not in ephys_report.QualityMetricReport():
                ephys_report.QualityMetricReport.populate(cluster_dropdown.value)

            firing_rate_plot = from_json(
                (ephys_report.QualityMetricReport & cluster_dropdown.value).fetch1(
                    "firing_rate_plot"
                )
            )

            presence_ratio_plot = from_json(
                (ephys_report.QualityMetricReport & cluster_dropdown.value).fetch1(
                    "presence_ratio_plot"
                )
            )

        else:
            qc_figs = QualityMetricFigs(cluster_dropdown)
            firing_rate_plot = qc_figs.empty_fig()
            presence_ratio_plot = qc_figs.empty_fig()

        with fwg.batch_update():
            fwg.data[0] = firing_rate_plot
            fwg.data[1] = presence_ratio_plot

    figure_output = wg.VBox(
        [fwg], layout=wg.Layout(width="95%", grid_area="figure_output")
    )
    figure_output.add_class("box_style")

    load_button.on_click(response)

    main_container = wg.GridBox(
        children=[
            title_button,
            cluster_dropdown,
            cutoff_dropdown,
            load_button,
            figure_output,
        ],
        layout=wg.Layout(
            grid_template_areas="""
            "title_button title_button title_button"
            "cluster_dropdown . load_button"
            "cutoff_dropdown . load_button"
            "figure_output figure_output figure_output"
            """
        ),
        grid_template_rows="auto auto auto auto",
        grid_template_columns="auto auto auto",
    )

    return main_container
