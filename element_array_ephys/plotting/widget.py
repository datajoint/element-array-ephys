import ipywidgets
from .. import ephys_report
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib
import typing as T
from IPython.display import display


def _drop_down_wg(
    dj_table, *, description="", width=100, description_width=100
) -> ipywidgets.Dropdown:
    return ipywidgets.Dropdown(
        options=dj_table.fetch("KEY", as_dict=True),
        description=description,
        disabled=False,
        layout=ipywidgets.Layout(
            width=f"{width}%",
            display="flex",
            flex_flow="row",
            justify_content="space-between",
            grid_area="processed_dropdown",
        ),
        style={"description_width": f"{description_width}px"},
    )


def plot_probe_level_report(probe_key: T.Dict[str, T.Any]) -> None:
    fig_name = (ephys_report.ProbeLevelReport & probe_key).fetch1("drift_map_plot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    img = mpimg.imread(fig_name)
    ax.imshow(img)
    ax.axis("off")
    plt.show()
    pathlib.Path(fig_name).unlink()


def probe_widget() -> ipywidgets.interactive:
    drop_down_wg = _drop_down_wg(
        ephys_report.ProbeLevelReport,
        description="Select Probe Insertion : ",
        width=85,
        description_width=150,
    )
    return ipywidgets.interactive(plot_probe_level_report, probe_key=drop_down_wg)


def plot_unit_level_report(unit_key: T.Dict[str, T.Any]) -> None:
    waveform_fig, autocorrelogram_fig, depth_waveform_fig = (
        ephys_report.UnitLevelReport & unit_key
    ).fetch1("waveform_plotly", "autocorrelogram_plotly", "depth_waveform_plotly")
    waveform_fig = go.FigureWidget(waveform_fig)
    autocorrelogram_fig = go.FigureWidget(autocorrelogram_fig)
    depth_waveform_fig = go.FigureWidget(depth_waveform_fig)

    display(
        ipywidgets.HBox(
            [ipywidgets.VBox([waveform_fig, autocorrelogram_fig]), depth_waveform_fig]
        )
    )


def unit_widget() -> ipywidgets.interactive:
    drop_down_wg = _drop_down_wg(
        ephys_report.UnitLevelReport & "cluster_quality_label='good'",
        description="Select Units : ",
        width=85,
        description_width=150,
    )
    return ipywidgets.interactive(plot_unit_level_report, unit_key=drop_down_wg)
