import logging
from functools import reduce

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.basedatatypes import BaseTraceType


class TraceManager:
    """
    Manages the content of a Trace based on spatially-aware downsampling.
    """

    def __init__(self, trace: BaseTraceType, buffer_ratio: float):
        self.max_points = 0
        self.trace = trace
        self.buffer_ratio = buffer_ratio
        self.trace_df, self.dimensions = self.extract_df(trace)
        self.visible_points = len(self.trace_df)

    def extract_df(self, trace: BaseTraceType) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract the data passed to the trace and return a pd.DataFrame.
        """
        trace_df = pd.DataFrame()
        axis_names = ["x", "y", "z", "lon", "lat"]

        for attr_name in dir(trace):
            if attr_name.startswith("_") or attr_name == "colorscale":
                continue
            attr = getattr(trace, attr_name)
            if isinstance(attr, (list | tuple | np.ndarray | pd.Series)):
                trace_df[attr_name] = pd.Series(attr)

        plot_dimensions = [d for d in axis_names if d in trace_df.columns]
        return trace_df, plot_dimensions

    def refresh(self, view_port: dict[str, int | float] | None = None):
        """
        Perform downsampling based on viewport with buffer and update the trace.
        """
        vp = view_port or {}
        dims = [d for d in self.dimensions if vp.get(f"{d}1", None) is not None]

        t_slice = self.trace_df if not dims else self._filter_by_viewport(vp, dims)
        self.visible_points = t_slice.shape[0]

        if self.visible_points > self.max_points:
            t_slice = self.spatial_downsample(t_slice)

        self.sliced_df = t_slice.columns.difference(["x_bin", "y_bin"])
        for col in t_slice.columns.difference(["x_bin", "y_bin"]):
            setattr(self.trace, col, t_slice[col].values)

    def _filter_by_viewport(self, view_port, dims):
        """
        Filter points within the expanded viewport (with buffer).
        """
        expanded_vp = {}
        for d in dims:
            d_min = view_port[f"{d}1"]
            d_max = view_port[f"{d}2"]
            range_size = d_max - d_min

            buffer_size = self.buffer_ratio * range_size
            expanded_vp[f"{d}1"] = d_min - buffer_size
            expanded_vp[f"{d}2"] = d_max + buffer_size

        filts = [
            (self.trace_df[d] >= expanded_vp[f"{d}1"]) &
            (self.trace_df[d] <= expanded_vp[f"{d}2"]) for d in dims
        ]
        filts = reduce(lambda x, y: x & y, filts)
        return self.trace_df.loc[filts]

    def spatial_downsample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        2D spatial downsampling
        """
        x_col, y_col = "x", "y"

        num_bins = int(np.sqrt(self.max_points))

        x_edges = np.linspace(data[x_col].min(), data[x_col].max(), num_bins + 1)
        y_edges = np.linspace(data[y_col].min(), data[y_col].max(), num_bins + 1)

        data.loc[:, "x_bin"] = np.digitize(data[x_col], bins=x_edges)
        data.loc[:, "y_bin"] = np.digitize(data[y_col], bins=y_edges)

        return data.groupby(["x_bin", "y_bin"], as_index=False).nth(0)


class DynamicPlot:
    """
    Wrapper for plotly's go.Figure object. Uses density based downsampling,
    preserving the ratio between traces.

    Parameters
    ----------
    fig : go.Figure
        Figure to dynamically downsample.
    resolution : int
        Amount of points to render (from all traces).
    buffer_ratio : float
        A buffer zone is added beyond the viewpoint to each side (top, bottom, lef, right)
        to smooth out the panning behavior.
        buffer_ratio is a factor by which the rendering area exceeds the viewpoint in each direction.
    """

    def __init__(self, fig: go.Figure, resolution: int = 100_000, buffer_ratio: float = 1):
        self.fig = fig
        self.resolution = resolution
        self.buffer_ratio = buffer_ratio

        self.trace_managers = [TraceManager(t, buffer_ratio=self.buffer_ratio) for t in fig.data]
        self.prev_vw = {}
        self._refresh_traces()

    def _set_scaled_max_points(self):
        """Scale the max_points for each trace based on the number of points in each trace."""
        total_visible_points = sum(tm.visible_points for tm in self.trace_managers)
        min_points_per_trace = 1

        for tm in self.trace_managers:
            tm.max_points = max(
                int((tm.visible_points / total_visible_points) * self.resolution),
                min_points_per_trace)
            # print(f"Trace {tm.trace["name"]}: max_points = {tm.max_points}")
        # print(f"Sum: {sum(tm.max_points for tm in self.trace_managers)}")

    def _refresh_traces(self, view_port: dict[str, float] | None = None):
        """
        Refresh and downsample the traces based on viewport.
        """
        view_port = view_port or {}
        for trace in self.trace_managers:
            trace.refresh(view_port)
        self._set_scaled_max_points()

    def refine_plot(self, relayout_data: dict[str, float]) -> go.Figure:
        """
        Refine plot based on relayout data from zoom or pan actions.

        Parameters
        ----------
        relayout_data : dict[str, float]
            data from dash's relayoutData callback (corners of viewpoint, dragmode changes)

        Returns
        -------
        go.Figure
            The downsampled and (optionally) cropped figure object.
        """
        rl = relayout_data or {}
        # print(rl)

        # Extract viewport info from relayout data
        if "xaxis.range[0]" in rl:
            vw = {
                "x1": rl.get("xaxis.range[0]"),
                "x2": rl.get("xaxis.range[1]"),
                "y1": rl.get("yaxis.range[0]"),
                "y2": rl.get("yaxis.range[1]"),
            }
        elif "dragmode" in rl:  # preserve dragmode and viewport coords on dragmode change
            self.fig.update_layout(dragmode=rl["dragmode"])
            vw = self.prev_vw
        else:
            vw = {}  # restore initial fig state

        self.prev_vw = vw
        self._refresh_traces(vw)

        if vw:  # viewport is smaller than what traces render because of the buffer
            self.fig.update_layout(xaxis_range=[vw["x1"], vw["x2"]],
                                   yaxis_range=[vw["y1"], vw["y2"]])
        else:
            self.fig.update_layout(xaxis=dict(range=None),
                                   yaxis=dict(range=None))
            self._refresh_traces(vw)

        return self.fig

    def show(self, *args, **kwargs):
        """
        Displays the plot by creating a dash app. Works inline in notebooks.

        All arguments are passed to :meth:`Dash.run_server`.
        Returns None.
        """
        print(" "*100)  # fix for Hydrogen
        app = Dash()
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        app.layout = html.Div([
            dcc.Graph(id="main_plot",
                      figure=self.fig, style={"height": "100%"})],
            style={"height": "100%"})

        app.callback(
            Output("main_plot", "figure"),
            [Input("main_plot", "relayoutData")],
        )(self.refine_plot)
        app.run_server(*args, **kwargs)


dd = DynamicPlot
