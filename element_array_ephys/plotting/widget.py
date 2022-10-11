from ipywidgets import widgets as wg
import element_array_ephys.ephys_acute as ephys
from element_array_ephys import ephys_report
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pathlib


def probe_widget(func):

    probe_dropdown = wg.Dropdown(
        options=ephys_report.ProbeLevelReport.fetch("KEY", as_dict=True),
        description="Select Probe Insertion : ",
        disabled=False,
        layout=wg.Layout(
            width="85%",
            display="flex",
            flex_flow="row",
            justify_content="space-between",
            grid_area="processed_dropdown",
        ),
        style={"description_width": "150px"},
    )

    return wg.interact(func, key=probe_dropdown)


def plot_probe_level_report(key: dict):
    fig_name = (ephys_report.ProbeLevelReport & key).fetch1("drift_map_plot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    img = mpimg.imread(fig_name)
    ax.imshow(img)
    ax.axis("off")
    plt.show()
    pathlib.Path(fig_name).unlink()


unit_dropdown = wg.Dropdown(
    options=(ephys.CuratedClustering.Unit).fetch("KEY", as_dict=True),
    description="Select Units : ",
    disabled=False,
    layout=wg.Layout(
        width="50%",
        display="flex",
        flex_flow="row",
        justify_content="space-between",
        grid_area="processed_dropdown",
    ),
    style={"description_width": "100px"},
)
