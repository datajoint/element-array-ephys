import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d
from .. import ephys_no_curation as ephys


class QualityMetricFigs(object):
    def __init__(self, key=None, **kwargs) -> None:
        self._key = key
        self._amplitude_cutoff_max = kwargs.get("amplitude_cutoff_maximum", None)
        self._presence_ratio_min = kwargs.get("presence_ratio_minimum", None)
        self._isi_violations_max = kwargs.get("isi_violations_maximum", None)
        self._units = pd.DataFrame()

    @property
    def units(self):
        assert self._key, "Must use key when retrieving units for QC figures"
        if self._units.empty:
            restrictions = ["TRUE"]
            if self._amplitude_cutoff_max:
                restrictions.append(f"amplitude_cutoff < {self._amplitude_cutoff_max}")
            if self._presence_ratio_min:
                restrictions.append(f"presence_ratio > {self._presence_ratio_min}")
            if self._isi_violations_max:
                restrictions.append(f"isi_violation < {self._isi_violations_max}")
            " AND ".join(restrictions)
            return (
                ephys.QualityMetrics
                * ephys.QualityMetrics.Cluster
                * ephys.QualityMetrics.Waveform
                & self._key
                & restrictions
            ).fetch(format="frame")
        return self._units

    def _plot_metric(
        self,
        data,
        bins,
        x_axis_label,
        scale=1,
        vline=None,
    ):
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=x_axis_label,
            template="plotly_dark",  # "simple_white",
            width=350 * scale,
            height=350 * scale,
            margin=dict(l=20 * scale, r=20 * scale, t=20 * scale, b=20 * scale),
            xaxis=dict(showgrid=False, zeroline=False, linewidth=2, ticks="outside"),
            yaxis=dict(showgrid=False, linewidth=0, zeroline=True, visible=False),
        )
        if data.isnull().all():
            return fig.add_annotation(text="No data available", showarrow=False)

        histogram, histogram_bins = np.histogram(data, bins=bins, density=True)

        fig.add_trace(
            go.Scatter(
                x=histogram_bins[:-1],
                y=gaussian_filter1d(histogram, 1),  # TODO: remove smoothing
                mode="lines",
                line=dict(color="rgb(0, 160, 223)", width=2 * scale),  # DataJoint Blue
                hovertemplate="%{x:.2f}<br>%{y:.2f}<extra></extra>",
            )
        )

        if vline:
            fig.add_vline(x=vline, line_width=2 * scale, line_dash="dash")

        return fig

    def empty_fig(self):  # TODO: Remove before submission?
        return self._plot_metric(
            pd.Series(["nan"]), np.linspace(0, 0, 0), "This fig left blank"
        )

    def firing_rate_plot(self):
        return self._plot_metric(
            np.log10(self.units["firing_rate"]),
            np.linspace(-3, 2, 100),  # If linear, use np.linspace(0, 50, 100)
            "log<sub>10</sub> firing rate (Hz)",
        )

    def presence_ratio_plot(self):
        return self._plot_metric(
            self.units["presence_ratio"],
            np.linspace(0, 1, 100),
            "Presence ratio",
            vline=0.9,
        )

    def amp_cutoff_plot(self):
        return self._plot_metric(
            self.units["amplitude_cutoff"],
            np.linspace(0, 0.5, 200),
            "Amplitude cutoff",
            vline=0.1,
        )

    def isi_violation_plot(self):
        return self._plot_metric(
            np.log10(self.units["isi_violation"] + 1e-5),  # Offset b/c log(0)
            np.linspace(-6, 2.5, 100),  # If linear np.linspace(0, 10, 200)
            "log<sub>10</sub> ISI violations",
            vline=np.log10(0.5),
        )

    def snr_plot(self):
        return self._plot_metric(self.units["snr"], np.linspace(0, 10, 100), "SNR")

    def iso_dist_plot(self):
        return self._plot_metric(
            self.units["isolation_distance"],
            np.linspace(0, 170, 50),
            "Isolation distance",
        )

    def d_prime_plot(self):
        return self._plot_metric(
            self.units["d_prime"], np.linspace(0, 15, 50), "d-prime"
        )

    def nn_hit_plot(self):
        return self._plot_metric(
            self.units["nn_hit_rate"],
            np.linspace(0, 1, 100),
            "Nearest-neighbors hit rate",
        )
