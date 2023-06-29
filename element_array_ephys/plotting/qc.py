import logging
import types

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger("datajoint")


class QualityMetricFigs(object):
    def __init__(
        self,
        ephys: types.ModuleType,
        key: dict = None,
        scale: float = 1,
        fig_width=800,
        amplitude_cutoff_maximum: float = None,
        presence_ratio_minimum: float = None,
        isi_violations_maximum: float = None,
        dark_mode: bool = False,
    ):
        """Initialize QC metric class

        Args:
            ephys (module): datajoint module with a QualityMetric table
            key (dict, optional): key from ephys.QualityMetric table. Defaults to None.
            scale (float, optional): Scale at which to render figure. Defaults to 1.4.
            fig_width (int, optional): Figure width in pixels. Defaults to 800.
            amplitude_cutoff_maximum (float, optional): Cutoff for unit amplitude in visualizations. Defaults to None.
            presence_ratio_minimum (float, optional): Cutoff for presence ratio in visualizations. Defaults to None.
            isi_violations_maximum (float, optional): Cutoff for isi violations in visualizations. Defaults to None.
            dark_mode (bool, optional): Set background to black, foreground white. Default False, black on white.
        """
        self._ephys = ephys
        self._key = key
        self._scale = scale
        self._plots = {}  # Empty default to defer set to dict property below
        self._fig_width = fig_width
        self._amplitude_cutoff_max = amplitude_cutoff_maximum
        self._presence_ratio_min = presence_ratio_minimum
        self._isi_violations_max = isi_violations_maximum
        self._dark_mode = dark_mode
        self._units = pd.DataFrame()  # Empty default
        self._x_fmt = dict(showgrid=False, zeroline=False, linewidth=2, ticks="outside")
        self._y_fmt = dict(showgrid=False, linewidth=0, zeroline=True, visible=False)
        self._no_data_text = "No data available"  # What to show when no data in table
        self._null_series = pd.Series(np.nan)  # What to substitute when no data

    @property
    def key(self) -> dict:
        """Key in ephys.QualityMetrics table"""
        return self._key

    @key.setter  # Allows `cls.property = new_item` notation
    def key(self, key: dict):
        """Use class_instance.key = your_key to reset key"""
        if key not in self._ephys.QualityMetrics.fetch("KEY"):
            # If not already full key, check if uniquely identifies entry
            key = (self._ephys.QualityMetrics & key).fetch1("KEY")
        self._key = key

    @key.deleter  # Allows `del cls.property` to clear key
    def key(self):
        """Use del class_instance.key to clear key"""
        logger.info("Cleared key")
        self._key = None

    @property
    def cutoffs(self) -> dict:
        """Amplitude, presence ratio, isi violation cutoffs"""
        return dict(
            amplitude_cutoff_maximum=self._amplitude_cutoff_max,
            presence_ratio_minimum=self._presence_ratio_min,
            isi_violations_maximum=self._isi_violations_max,
        )

    @cutoffs.setter
    def cutoffs(self, cutoff_dict):
        """Use class_instance.cutoffs = dict(var=cutoff) to adjust cutoffs

        Args:
            cutoff_dict (kwargs): Cutoffs to adjust: amplitude_cutoff_maximum,
                presence_ratio_minimum, and/or isi_violations_maximum
        """
        self._amplitude_cutoff_max = cutoff_dict.get(
            "amplitude_cutoff_maximum", self._amplitude_cutoff_max
        )
        self._presence_ratio_min = cutoff_dict.get(
            "presence_ratio_minimum", self._presence_ratio_min
        )
        self._isi_violations_max = cutoff_dict.get(
            "isi_violations_maximum", self._isi_violations_max
        )
        _ = self.units

    @property
    def units(self) -> pd.DataFrame:
        """Pandas dataframe of QC metrics"""
        if not self._key:
            return self._null_series

        if self._units.empty:
            restrictions = ["TRUE"]
            if self._amplitude_cutoff_max:
                restrictions.append(f"amplitude_cutoff < {self._amplitude_cutoff_max}")
            if self._presence_ratio_min:
                restrictions.append(f"presence_ratio > {self._presence_ratio_min}")
            if self._isi_violations_max:
                restrictions.append(f"isi_violation < {self._isi_violations_max}")
            " AND ".join(restrictions)  # Build restriction from cutoffs
            return (
                self._ephys.QualityMetrics
                * self._ephys.QualityMetrics.Cluster
                * self._ephys.QualityMetrics.Waveform
                & self._key
                & restrictions
            ).fetch(format="frame")

        return self._units

    def _format_fig(
        self, fig: go.Figure = None, scale: float = None, ratio: float = 1.0
    ) -> go.Figure:
        """Return formatted figure or apply formatting to existing figure

        Args:
            fig (go.Figure, optional): Apply formatting to this plotly graph object
                Figure to apply formatting. Defaults to empty.
            scale (float, optional): Scale to render figure. Defaults to scale from
                class init, 1.
            ratio (float, optional): Figure aspect ratio width/height. Defaults to 1.

        Returns:
            go.Figure: Formatted figure
        """
        if not fig:
            fig = go.Figure()
        if not scale:
            scale = self._scale

        width = self._fig_width * scale

        return fig.update_layout(
            template="plotly_dark" if self._dark_mode else "simple_white",
            width=width,
            height=width / ratio,
            margin=dict(l=20 * scale, r=20 * scale, t=40 * scale, b=40 * scale),
            showlegend=False,
        )

    def _empty_fig(
        self, text="Select a key to visualize QC metrics", scale=None
    ) -> go.Figure:
        """Return figure object for when no key is provided"""
        if not scale:
            scale = self._scale

        return (
            self._format_fig(scale=scale)
            .add_annotation(text=text, showarrow=False)
            .update_layout(xaxis=self._y_fmt, yaxis=self._y_fmt)
        )

    def _plot_metric(
        self,
        data: pd.DataFrame,
        bins: np.ndarray,
        scale: float = None,
        fig: go.Figure = None,
        **trace_kwargs,
    ) -> go.Figure:
        """Plot histogram using bins provided

        Args:
            data (pd.DataFrame): Data to be plotted, from QC metric
            bins (np.ndarray): Array of bins to use for histogram
            scale (float, optional): Scale to render figure. Defaults to scale from
                class initialization.
            fig (go.Figure, optional): Add trace to this figure. Defaults to empty
                formatted figure.

        Returns:
            go.Figure: Histogram plot
        """
        if not scale:
            scale = self._scale
        if not fig:
            fig = self._format_fig(scale=scale)

        if not data.isnull().all():
            histogram, histogram_bins = np.histogram(data, bins=bins, density=True)
        else:
            # To quiet divide by zero error when no data
            histogram, histogram_bins = np.ndarray(0), np.ndarray(0)

        return fig.add_trace(
            go.Scatter(
                x=histogram_bins[:-1],
                y=gaussian_filter1d(histogram, 1),
                mode="lines",
                line=dict(color="rgb(0, 160, 223)", width=2 * scale),  # DataJoint Blue
                hovertemplate="%{x:.2f}<br>%{y:.2f}<extra></extra>",
            ),
            **trace_kwargs,
        )

    def get_single_fig(self, fig_name: str, scale: float = None) -> go.Figure:
        """Return a single figure of the plots listed in the plot_list property

        Args:
            fig_name (str): Name of figure to be rendered
            scale (float, optional): Scale to render fig. Defaults to scale at class init, 1.

        Returns:
            go.Figure: Histogram plot
        """
        if not self._key:
            return self._empty_fig()
        if not scale:
            scale = self._scale

        fig_dict = self.plots.get(fig_name, dict()) if self._key else dict()
        data = fig_dict.get("data", self._null_series)
        bins = fig_dict.get("bins", np.linspace(0, 0, 0))
        vline = fig_dict.get("vline", None)

        if data.isnull().all():
            return self._empty_fig(text=self._no_data_text)

        fig = (
            self._plot_metric(data=data, bins=bins, scale=scale)
            .update_layout(xaxis=self._x_fmt, yaxis=self._y_fmt)
            .update_layout(  # Add title
                title=dict(text=fig_dict.get("xaxis", " "), xanchor="center", x=0.5),
                font=dict(size=12 * scale),
            )
        )

        if vline:
            fig.add_vline(x=vline, line_width=2 * scale, line_dash="dash")

        return fig

    def get_grid(self, n_columns: int = 4, scale: float = 1.0) -> go.Figure:
        """Plot grid of histograms as subplots in go.Figure using n_columns

        Args:
            n_columns (int, optional): Number of column in grid. Defaults to 4.
            scale (float, optional): Scale to render fig. Defaults to scale at class init, 1.

        Returns:
            go.Figure: grid of available plots
        """
        from plotly.subplots import make_subplots

        if not self._key:
            return self._empty_fig()
        if not scale:
            scale = self._scale

        n_rows = int(np.ceil(len(self.plots) / n_columns))

        fig = self._format_fig(
            fig=make_subplots(
                rows=n_rows,
                cols=n_columns,
                shared_xaxes=False,
                shared_yaxes=False,
                vertical_spacing=(0.5 / n_rows),
            ),
            scale=scale,
            ratio=(n_columns / n_rows),
        ).update_layout(  # Global title
            title=dict(text="Histograms of Quality Metrics", xanchor="center", x=0.5),
            font=dict(size=12 * scale),
        )

        for idx, plot in enumerate(self._plots.values()):  # Each subplot
            this_row = int(np.floor(idx / n_columns) + 1)
            this_col = idx % n_columns + 1
            data = plot.get("data", self._null_series)
            vline = plot.get("vline", None)
            if data.isnull().all():
                vline = None  # If no data, don't want vline either
                fig["layout"].update(
                    annotations=[
                        dict(
                            xref=f"x{idx+1}",
                            yref=f"y{idx+1}",
                            text=self._no_data_text,
                            showarrow=False,
                        ),
                    ]
                )
            fig = self._plot_metric(  # still need to plot empty to cal y_vals min/max
                data=data,
                bins=plot["bins"],
                fig=fig,
                row=this_row,
                col=this_col,
                scale=scale,
            )
            fig.update_xaxes(
                title=dict(text=plot["xaxis"], font_size=11 * scale),
                row=this_row,
                col=this_col,
            )
            if vline:
                y_vals = fig.to_dict()["data"][idx]["y"]
                fig.add_shape(  # Add overlay WRT whole fig
                    go.layout.Shape(
                        type="line",
                        yref="paper",
                        xref="x",  # relative to subplot x
                        x0=vline,
                        y0=min(y_vals),
                        x1=vline,
                        y1=max(y_vals),
                        line=dict(width=2 * scale),
                    ),
                    row=this_row,
                    col=this_col,
                )

        return fig.update_xaxes(**self._x_fmt).update_yaxes(**self._y_fmt)

    @property
    def plot_list(self):
        """List of plots that can be rendered individually by name or as grid"""
        if not self._plots:
            _ = self.plots
        return [plot for plot in self._plots]

    @property
    def plots(self):
        if not self._plots:
            self._plots = {
                "firing_rate": {  # If linear, use np.linspace(0, 50, 100)
                    "xaxis": "Firing rate (log<sub>10</sub> Hz)",
                    "data": np.log10(self.units.get("firing_rate", self._null_series)),
                    "bins": np.linspace(-3, 2, 100),
                },
                "presence_ratio": {
                    "xaxis": "Presence ratio",
                    "data": self.units.get("presence_ratio", self._null_series),
                    "bins": np.linspace(0, 1, 100),
                    "vline": 0.9,
                },
                "amp_cutoff": {
                    "xaxis": "Amplitude cutoff",
                    "data": self.units.get("amplitude_cutoff", self._null_series),
                    "bins": np.linspace(0, 0.5, 200),
                    "vline": 0.1,
                },
                "isi_violation": {  # If linear bins(0, 10, 200). Offset b/c log(0) null
                    "xaxis": "ISI violations (log<sub>10</sub>)",
                    "data": np.log10(
                        self.units.get("isi_violation", self._null_series) + 1e-5
                    ),
                    "bins": np.linspace(-6, 2.5, 100),
                    "vline": np.log10(0.5),
                },
                "snr": {
                    "xaxis": "SNR",
                    "data": self.units.get("snr", self._null_series),
                    "bins": np.linspace(0, 10, 100),
                },
                "iso_dist": {
                    "xaxis": "Isolation distance",
                    "data": self.units.get("isolation_distance", self._null_series),
                    "bins": np.linspace(0, 170, 50),
                },
                "d_prime": {
                    "xaxis": "d-prime",
                    "data": self.units.get("d_prime", self._null_series),
                    "bins": np.linspace(0, 15, 50),
                },
                "nn_hit": {
                    "xaxis": "Nearest-neighbors hit rate",
                    "data": self.units.get("nn_hit_rate", self._null_series),
                    "bins": np.linspace(0, 1, 100),
                },
            }
        return self._plots

    @plots.setter
    def plots(self, new_plot_dict: dict):
        """Adds or updates plot item in the set to be rendered.

        plot items are structured as followed: dict with name key, embedded dict with
            xaxis: string x-axis label
            data: pandas dataframe to be plotted
            bins: numpy ndarray of bin cutoffs for histogram
        """
        _ = self.plots
        [self._plots.update({k: v}) for k, v in new_plot_dict.items()]

    def remove_plot(self, plot_name):
        """Removes an item from the set of plots"""
        _ = self._plots.pop(plot_name)
