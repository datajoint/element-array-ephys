import ipywidgets as widgets
from .. import ephys_report
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib
import typing as T
from IPython.display import display
from .. import ephys_no_curation as ephys
from skimage import io


# Widgets
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

# Update shank & unit dropdown according to probe change
def update_shank_options(change):
    probe_key = change.new
    shank_dropdown_wg.options = (ephys_report.ProbeLevelReport & probe_key.value).fetch(
        "shank"
    )


def update_unit_dropdown(change):
    probe_key = change.new
    unit_dropdown_wg.options = (
        (
            (ephys_report.UnitLevelReport & probe_key.value)
            & "cluster_quality_label='good'"
        ).fetch("unit"),
    )


probe_dropdown_wg.observe(update_shank_options, "value")
probe_dropdown_wg.observe(update_unit_dropdown, "value")


def plot_probe_level_report(probe_key: T.Dict[str, T.Any]) -> None:
    fig_name = (ephys_report.ProbeLevelReport & probe_key).fetch1("drift_map_plot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    img = mpimg.imread(fig_name)
    ax.imshow(img)
    ax.axis("off")
    plt.show()
    pathlib.Path(fig_name).unlink()


def plot_unit_level_report(unit_key: T.Dict[str, T.Any]) -> None:
    waveform_fig, autocorrelogram_fig, depth_waveform_fig = (
        ephys_report.UnitLevelReport & unit_key
    ).fetch1("waveform_plotly", "autocorrelogram_plotly", "depth_waveform_plotly")
    waveform_fig = go.FigureWidget(waveform_fig)
    autocorrelogram_fig = go.FigureWidget(autocorrelogram_fig)
    depth_waveform_fig = go.FigureWidget(depth_waveform_fig)

    display(
        widgets.HBox(
            [widgets.VBox([waveform_fig, autocorrelogram_fig]), depth_waveform_fig]
        )
    )
