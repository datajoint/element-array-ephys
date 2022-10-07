from ipywidgets import widgets as wg
import element_array_ephys.ephys_acute as ephys
from element_array_ephys import probe, report


probe_dropdown = wg.Dropdown(
    options=report.ProbeLevelReport.fetch("KEY", as_dict=True),
    description="Select Probe Insertion : ",
    disabled=False,
    layout=wg.Layout(
        width="60%",
        display="flex",
        flex_flow="row",
        justify_content="space-between",
        grid_area="processed_dropdown",
    ),
    style={"description_width": "150px"},
)


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
