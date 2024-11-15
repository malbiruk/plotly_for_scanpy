import numpy as np
import one_light_template  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde, iqr
from webcolors import hex_to_rgb, name_to_rgb


def _create_array_dict(data, category_col, data_col,
                       normalize_each, scaling_factor):
    array_dict = {}
    target_area = (data[data_col].max() - data[data_col].min()) * 0.25
    for category in list(data[category_col].unique()):
        array_dict[f"x_{category}"] = data[data[category_col] == category][data_col]
        array_dict[f"y_{category}"] = data[data[category_col] == category]["count"]
        if normalize_each:
            min_count = array_dict[f"y_{category}"].min()
            max_count = array_dict[f"y_{category}"].max()
            current_area = np.trapz(array_dict[f"y_{category}"], array_dict[f"x_{category}"])
            normalization_factor = target_area / current_area
            array_dict[f"y_{category}_normalized"] = array_dict[f"y_{
                category}"] * normalization_factor * scaling_factor

        else:
            min_count = data["count"].min()
            max_count = data["count"].max()
            array_dict[f"y_{category}_normalized"] = (array_dict[f"y_{category}"] - min_count) \
                / (max_count - min_count) * scaling_factor
    return array_dict


def create_histogram(data: pd.DataFrame,
                     category_col: str,
                     data_col: str,
                     bin_width=None,
                     n_bins=None,
                     ) -> pd.DataFrame:

    data = data.dropna(subset=data_col)

    if np.issubdtype(data[data_col].dtype, np.datetime64):
        data[data_col] = pd.to_datetime(data[data_col])

        if not n_bins:
            if not bin_width:
                # Default bin width: 1 day if unspecified
                bin_width = "1D"

            # Generate bins based on bin_width or calculate bin range
            if isinstance(bin_width, str):
                bins = pd.date_range(data[data_col].min(), data[data_col].max(), freq=bin_width)
            else:
                # If bin_width is a timedelta (e.g., pd.Timedelta(days=7))
                bins = pd.date_range(data[data_col].min(),
                                     data[data_col].max() + bin_width, freq=bin_width)

            n_bins = len(bins) - 1

        data["bin"] = pd.cut(data[data_col], bins=bins, right=False)

    else:
        if not n_bins:
            if not bin_width:
                bin_width = 2 * iqr(data[data_col]) * len(data[data_col]) ** (-1/3)
            n_bins = int((data[data_col].max() - data[data_col].min()) / bin_width)

        bins = np.linspace(data[data_col].min(), data[data_col].max(), n_bins + 1)

        data["bin"] = pd.cut(
            data[data_col],
            bins=bins,
            right=False,
        )

    binned_data = (
        data.groupby([category_col, "bin"])
        .size()
        .reset_index(name="count")
    )

    if np.issubdtype(data[data_col].dtype, np.datetime64):
        binned_data[data_col] = pd.to_datetime(binned_data["bin"].apply(lambda x: x.mid))
    else:
        binned_data[data_col] = binned_data["bin"].apply(lambda x: x.mid).astype(float)

    return binned_data


def create_kde_lines(data: pd.DataFrame,
                     category_col: str,
                     data_col: str,
                     n_points: int = 100,
                     bandwidth: float | None = None) -> pd.DataFrame:

    data = data.dropna(subset=[data_col])

    kde_lines = []

    for category, group in data.groupby(category_col):
        x = group[data_col].to_numpy()
        if len(x) < 2:
            continue

        if np.issubdtype(data[data_col].dtype, np.datetime64):
            x = (pd.to_datetime(x) - pd.Timestamp("1970-01-01")) / \
                pd.Timedelta(seconds=1)  # Convert datetime to numeric

        kde = gaussian_kde(x, bw_method=bandwidth)

        x_min, x_max = x.min(), x.max()
        x_grid = np.linspace(x_min, x_max, n_points)

        y_grid = kde(x_grid)

        if np.issubdtype(data[data_col].dtype, np.datetime64):
            x_grid = pd.to_datetime(x_grid, unit="s")

        kde_lines.append(pd.DataFrame({
            category_col: category,
            data_col: x_grid,
            "density": y_grid,
        }))

    return pd.concat(kde_lines, ignore_index=True)


def create_outline(x: np.array, y: np.array, x_diff: np.array) -> tuple[np.array]:
    x_outline = np.repeat(x - x_diff / 2, 2)[1:]
    x_outline = np.concatenate([x_outline, [x.to_numpy()[-1] + x_diff[-1] / 2]])
    y_outline = []
    for i in range(len(x)):
        y_outline.extend([y[i], y[i]])

    return x_outline, np.array(y_outline)


def color_to_rgb(color: str) -> tuple:
    if not isinstance(color, str):
        return color

    if color.startswith("#"):
        return (hex_to_rgb(color))

    return name_to_rgb(color)


def ridgeline(
        data: pd.DataFrame,
        *,
        category_col: str,
        data_col: str = "Data",
        stats_col: str | None = None,
        normalize_each: bool = True,
        scaling_factor: float = 1.75,
        edgecolor: str | None = None,
        colorway: list | None = None,
        ridgetype: str = "lines",
        smoothing: float = 0.9,
        opacity: float = 0.5,
        bin_width=None,
        n_bins=None,
        n_points: int = 100,
        hover_stats: bool = True,
        jitter: bool = False,
        jitter_size: float = 3,
        jitter_strength: float = 0.5,
        bandwidth: float | None = None) -> go.Figure:
    if ridgetype not in ["lines", "bins"]:
        raise ValueError(f"Type should be 'lines' or 'bins'. recieved {ridgetype}")

    initial_data = data
    stats_col_init = stats_col
    if not stats_col:
        if ridgetype == "bins":
            data = create_histogram(data, category_col, data_col, bin_width, n_bins)
            stats_col = "count"
        elif ridgetype == "lines":
            data = create_kde_lines(data, category_col, data_col, n_points, bandwidth)
            smoothing = 0
            stats_col = "density"

    fig = go.Figure()
    data = data.rename(columns={stats_col: "count", data_col: "Data"})
    array_dict = _create_array_dict(data, category_col, "Data",
                                    normalize_each, scaling_factor)
    categories_list = list(data[category_col].unique())
    for index, category in enumerate(categories_list):
        x = array_dict[f"x_{category}"]
        y = array_dict[f"y_{category}_normalized"].to_numpy()
        color = (colorway[index] if colorway
                 else pio.templates[pio.templates.default].layout.colorway[index])

        x_diff = np.diff(x)
        x_diff = np.concatenate(([x_diff[0]], x_diff))

        supp_x_small = (
            [array_dict[f"x_{category}"].min() - x_diff[-1] / 2,
                array_dict[f"x_{category}"].max() + x_diff[-1] / 2]
            if ridgetype == "bins" else
            [array_dict[f"x_{category}"].min(), array_dict[f"x_{category}"].max()]
        )

        supp_x_long = (
            [data["Data"].min() - x_diff[-1] / 2, data["Data"].max() + x_diff[-1] / 2]
            if ridgetype == "bins" else
            [data["Data"].min(), data["Data"].max()]
        )

        x, y = (x, y) if ridgetype == "lines" else create_outline(x, y, x_diff)

        # small supporting lines for drawing areas
        fig.add_trace(go.Scatter(
            x=supp_x_small,
            y=np.full(2, len(categories_list)-1-index),
            mode="lines",
            line_width=1,
            line_color="#eaeaea"))

        # drawing lines and areas
        fig.add_trace(go.Scatter(
            x=x,
            y=y+len(categories_list)-index-1,
            mode="lines",
            line_shape="spline",
            line_smoothing=smoothing if ridgetype == "lines" else 0,
            fillcolor=f"rgba{(*color_to_rgb(color), opacity)}",
            fill="tonexty",
            line=dict(color=edgecolor if edgecolor else color, width=2),
            showlegend=False,
            hoverinfo="skip"))

        # long supporting lines
        fig.add_trace(go.Scatter(
            x=supp_x_long,
            y=np.full(2, len(categories_list)-1-index),
            mode="lines",
            line_width=1,
            line_color="#eaeaea"))

        jitter_x = initial_data[initial_data[category_col] == category][data_col]
        if jitter and not stats_col_init:
            jitter_amount = np.random.uniform(0, jitter_strength, len(jitter_x))
            jitter_y = np.repeat(len(categories_list)-index-1, len(jitter_x)) + jitter_amount
            fig.add_trace(go.Scatter(
                x=jitter_x,
                y=jitter_y,
                mode="markers",
                marker_size=jitter_size,
                marker_color=edgecolor if edgecolor else color,
                name=f"{category}",
                hovertemplate="{{Data}}: %{x}".replace("{{Data}}", data_col)))

        hovertemplate = (
            "{{Data}}: %{x}<br>"
            "{{Duration}}: %{customdata}"
            .replace("{{Data}}", data_col)
            .replace("{{Duration}}", stats_col) if not hover_stats else
            "{{Data}}: %{x}<br>"
            "{{Duration}}: %{customdata}<br><br>"
            "max: {{max}}<br>"
            "upper fence: {{upper_fence}}<br>"
            "q3: {{q3}}<br>"
            "median: {{median}}<br>"
            "mean: {{mean}}<br>"
            "q1: {{q1}}<br>"
            "lower fence: {{lower_fence}}<br>"
            "min: {{min}}"
            .replace("{{Data}}", data_col)
            .replace("{{Duration}}", stats_col)
            .replace("{{max}}", str(jitter_x.max()))
            .replace("{{upper_fence}}", str(jitter_x.quantile(0.75)
                     + 1.5 * (jitter_x.quantile(0.75) - jitter_x.quantile(0.25))))
            .replace("{{q3}}", str(jitter_x.quantile(0.75)))
            .replace("{{median}}", str(jitter_x.median()))
            .replace("{{mean}}", str(jitter_x.mean()))
            .replace("{{q1}}", str(jitter_x.quantile(0.25)))
            .replace("{{lower_fence}}", str(jitter_x.quantile(0.25)
                     - 1.5 * (jitter_x.quantile(0.75) - jitter_x.quantile(0.25))))
            .replace("{{min}}", str(jitter_x.min()))
        )
        customdata = array_dict[f"y_{category}"]

        # invisible trace with hoverinfo
        fig.add_trace(go.Scatter(
            x=array_dict[f"x_{category}"],
            y=array_dict[f"y_{category}_normalized"].to_numpy()+len(categories_list)-1-index,
            opacity=0,
            marker_color=color,
            name=f"{category}",
            hovertemplate=hovertemplate,
            customdata=customdata),
        )

        # y-labels
        fig.add_annotation(
            x=-0.025,
            y=len(categories_list) - index - 1,
            xref="paper",
            xanchor="right",
            text=f"{category}",
            showarrow=False,
            yshift=4,
            yref="y",
            name="ytitles",
        )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=data_col, showline=False)
    fig.update_yaxes(showticklabels=False, showline=False, zeroline=False)
    return fig
