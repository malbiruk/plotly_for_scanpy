"""
This module provides plotting functions for AnnData objects utilizing plotly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from anndata import AnnData
from plotly.subplots import make_subplots
from plotly_for_scanpy import one_light_template  # noqa: F401


def qc_metrics(
    adata: AnnData,
    qc_vars: str | list | None = None,
    ncols: int = 3,
    *,
    width: int | None = None,
    height: int | None = None,
    quantile: float = 1.0,
    template: str | None = None,
    horizontal_spacing: float = 0.075,
    vertical_spacing: float = 0.1,
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Calculate and create histograms of QC metrics:
        N counts per cell, N genes per cell, % of mitochondrial expression per cell

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    width : int | None
        Width of the figure in pixels.
    height : int | None
        Height of the figure in pixels.
    layout : str
        Layout orientation ('vertical' or 'horizontal').
    quantile: float
        PLot data only of the first quantile (if specified).
    template : str | dict
        Plotly template name or template dict.
    mt_col : str
        Column name in data.var containing bool info if gene is mitochondrial
        (the column will be cerated if it doesn't exist).
    return_fig : bool
        If True, return the figure instead of displaying it.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    if template is None:
        template = pio.templates.default

    if qc_vars:
        qc_vars = [qc_vars] if isinstance(qc_vars, str) else list(qc_vars)

    num_plots = 2 + len(qc_vars) if qc_vars else 2
    cols = min(num_plots, ncols)
    rows = int(np.ceil(num_plots / cols))

    subplot_titles = (["N of UMI per cell", "N of genes per cell"] +
                      [f"% of {qc_var} expression per cell" for qc_var in qc_vars] if qc_vars
                      else ["N of UMI per cell", "N of genes per cell"])

    fig = make_subplots(rows=rows, cols=cols,
                        horizontal_spacing=horizontal_spacing,
                        vertical_spacing=vertical_spacing,
                        subplot_titles=subplot_titles)

    all_metrics = (["total_counts", "n_genes_by_counts"]
                   + [f"pct_counts_{qc_var}" for qc_var in qc_vars] if qc_vars
                   else ["total_counts", "n_genes_by_counts"])

    for c, metric in enumerate(all_metrics):
        row = (c // cols) + 1
        col = (c % cols) + 1
        threshold = np.percentile(adata.obs[metric], quantile * 100)
        fig.add_trace(go.Histogram(x=adata.obs[metric][adata.obs[metric] <= threshold]),
                      row=row, col=col)

    for row in range(1, rows+1):
        for col in range(1, cols+1):
            fig.update_yaxes(showticklabels=False, ticks="", showline=False, row=row, col=col)

    fig.update_layout(showlegend=False,
                      template=template, width=width, height=height)
    fig.update_annotations(font={"family": "Serif"})

    if return_fig:
        return fig

    fig.show()
    return None


def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    """
    Get array for basis from anndata. Just tries to add 'X_'.
    """
    if basis in adata.obsm:
        return adata.obsm[basis]
    if f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    err_msg = f"Could not find '{basis}' or 'X_{basis}' in .obsm"
    raise KeyError(err_msg)


def _prepare_dimension_dataframes(adata, basis, dimensions, last_color_col, groups):
    """
    Prepare DataFrames for each dimension pair.
    """
    bas = basis[2:].upper() if basis.startswith("X_") else basis.upper()
    dfs_for_plot = []

    for dim_pair in dimensions:
        new_cols_df = pd.DataFrame(
            _get_basis(adata, basis)[:, dim_pair],
            columns=[f"{bas}{dim_pair[0]+1}", f"{bas}{dim_pair[1]+1}"],
        )
        df_for_plot = pd.concat(
            [adata.obs.reset_index(), new_cols_df],
            axis=1,
        ).set_index(adata.obs_names)

        df_for_plot = (df_for_plot[df_for_plot[last_color_col].isin(groups)]
                       if groups else df_for_plot)

        dfs_for_plot.append(df_for_plot)

    return dfs_for_plot, bas


def _add_color_data(adata, dfs_for_plot, color):
    """
    Add color data to DataFrames.
    """
    for df_for_plot in dfs_for_plot:
        if color == [""]:
            df_for_plot[""] = pd.Categorical([""] * len(df_for_plot))
        else:
            for color_col in color:
                err_msg = f"Could not find key {color_col} in .var_names or .obs.columns."
                if color_col not in adata.obs.columns:
                    if color_col in adata.var_names:
                        df_for_plot[color_col] = adata[:,
                                                       adata.var_names == color_col].X.toarray().T[0]
                    else:
                        raise KeyError(err_msg)

    return color


def _add_trace_to_figure(fig, trace, row, col, counter, color_col, dim_pair,
                         *, group_legends: bool):
    """
    Add a trace to the figure with proper formatting.
    """
    legendgroup_name = f"{color_col} {dim_pair}"
    fig.add_trace(trace, row=row, col=col)
    if group_legends:
        fig.update_traces(
            legendgrouptitle=dict(text=legendgroup_name),
            legendgroup=legendgroup_name,
            marker=dict(coloraxis=f"coloraxis{counter+1}"),
            row=row,
            col=col,
        )
    else:
        fig.update_traces(
            marker=dict(coloraxis=f"coloraxis{counter+1}"),
            row=row,
            col=col,
        )
    fig.update_layout({f"coloraxis{counter+1}": {"showscale": False}})


def _update_figure_layout(fig, template, marker_size, width, height, title):
    """
    Update the final figure layout.
    """
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        showlegend=True,
        template=template,
        legend_groupclick="toggleitem",
        margin={"pad": 20},
        width=width,
        height=height,
        title=title,
    )
    fig.update_yaxes(showticklabels=False, zeroline=False, ticks="")
    fig.update_xaxes(showticklabels=False, zeroline=False, ticks="")
    fig.update_annotations(font={"family": "Serif"})


def _calculate_centroids(df_for_plot, color_col, x_col, y_col):
    """
    Calculate centroids of each group for a given color_col.
    """
    return df_for_plot.groupby(color_col, observed=False)[[x_col, y_col]].mean()


def _add_annotations(fig, centroids, row, col, annotations_font):
    """
    Add annotations (category labels) at centroids.
    """
    for cat, coords in centroids.iterrows():
        fig.add_annotation(
            x=coords.iloc[0],
            y=coords.iloc[1],
            text=str(cat),
            showarrow=False,
            xref=f"x{col}",
            yref=f"y{row}",
            font=annotations_font)


def embedding(adata: AnnData,
              basis: str,
              *,
              marker_size: float = 5,
              template: str | None = None,
              dimensions: tuple[int, int] | list[tuple[int, int]] = (0, 1),
              color: str | list[str] | None = None,
              groups: list[str] | None = None,
              annotations: bool = False,
              annotations_font: dict | None = None,
              ncols: int = 3,
              horizontal_spacing: float = 0.1,
              vertical_spacing: float = 0.15,
              opacity: float = 0.25,
              width: int | None = None,
              height: int | None = None,
              subtitles: str | list[str] | None = None,
              title: str | None = None,
              use_scanpy_colors: bool | None = None,
              return_fig: bool = False,
              _pca_annotate_variances: bool = False,
              **kwargs,
              ) -> go.Figure | None:
    """
    Create embedding plots with multiple dimensions and colors.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str
        Basis for dimensional reduction (e.g., 'X_umap', 'X_tsne', 'X_pca').
    marker_size : float
        Size of markers in scatter plots.
    template : str | dict
        Plotly template name or template dict.
    dimensions : tuple[int, int] | list[tuple[int, int]]
        Dimensions to plot, either single tuple or list of tuples.
    color : str | list[str] | None
        Column name(s) to use for coloring.
    groups: list[str] | None
        Show only these groups from the last color in scatter plots.
    annotations: bool
        Show each group name in their centroids in scatter plots.
    annotations_font: dict
        Font parameters of annotations.
    ncols : int
        Maximum number of columns in subplot grid.
    horizontal_spacing : float
        Spacing between subplot columns.
    vertical_spacing : float
        Spacing between subplot rows.
    opacity : float
        Opacity of markers (0 to 1).
    width : int | None
        Width of figure in pixels.
    height : int | None
        Height of figure in pixels.
    subtitles : str | list[str] | None
        List of subtitles to use, color by default.
    title : str | None
        Common title for the whole figure.
    use_scanpy_colors : bool
        If True, use color mapping from adata.uns
    return_fig : bool
        If True, return the figure instead of displaying it.
    **kwargs
        Additional keyword arguments passed to px.scatter.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    template = template or pio.templates.default

    # Prepare data
    if isinstance(dimensions, tuple):
        dimensions = [dimensions]
    if color is None:
        color = [""]
    else:
        color = [color] if isinstance(color, str) else list(color)

    last_color_col = color[-1]
    dfs_for_plot, bas = _prepare_dimension_dataframes(
        adata, basis, dimensions, last_color_col, groups)
    color = _add_color_data(adata, dfs_for_plot, color)

    # Calculate layout
    num_plots = len(color) * len(dimensions)
    group_legends = num_plots > 1
    cols = min(num_plots, ncols)
    rows = int(np.ceil(num_plots / cols))

    # Create figure
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(subtitles) if subtitles is not None else color * len(dimensions),
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # Add subplots
    counter = 0
    for dim_pair, df_for_plot in zip(dimensions, dfs_for_plot):
        for color_col in color:
            x_col = f"{bas}{dim_pair[0]+1}"
            y_col = f"{bas}{dim_pair[1]+1}"

            color_discrete_map = dict(
                zip(adata.obs[color_col].cat.categories,
                    adata.uns[f"{color_col}_colors"])) if use_scanpy_colors else None

            if adata.obs[color_col].dtype.name == "category":
                categories = adata.obs[color_col].cat.categories
            else:
                categories = adata.obs[color_col].unique()

            px_fig = px.scatter(
                df_for_plot,
                x=x_col,
                y=y_col,
                template=template,
                color=color_col,
                opacity=opacity,
                category_orders={color_col: categories},
                color_discrete_map=color_discrete_map,
                **kwargs)

            for trace in px_fig["data"]:
                row = (counter // cols) + 1
                col = (counter % cols) + 1
                _add_trace_to_figure(fig, trace, row, col, counter,
                                     color_col, dim_pair, group_legends=group_legends)

            if annotations:
                centroids = _calculate_centroids(df_for_plot, color_col, x_col, y_col)
                _add_annotations(fig, centroids, row, col, annotations_font)

            if _pca_annotate_variances:
                var_x = round(adata.uns["pca"]["variance_ratio"][dim_pair[0]] * 100, 2)
                var_y = round(adata.uns["pca"]["variance_ratio"][dim_pair[1]] * 100, 2)
                fig.update_xaxes(title_text=f"{x_col} ({var_x}%)", row=row, col=col)
                fig.update_yaxes(title_text=f"{y_col} ({var_y}%)", row=row, col=col)
            else:
                fig.update_xaxes(title_text=x_col, row=row, col=col)
                fig.update_yaxes(title_text=y_col, row=row, col=col)
            counter += 1

    # Update final layout
    _update_figure_layout(fig, template, marker_size, width, height, title)

    if return_fig:
        return fig
    fig.show()
    return None


def pca(adata: AnnData, *,
        annotate_var_explained: bool = True,
        **kwargs,
        ) -> go.Figure | None:
    """
    Scatter plot in PCA coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    annotate_var_explained : bool
        Whether to annotate the explained variance on axis names
    **kwargs
        Additional keyword arguments passed to embedding.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    return embedding(adata, basis="pca", _pca_annotate_variances=annotate_var_explained, **kwargs)


def umap(adata: AnnData, **kwargs):
    """
    Scatter plot in UMAP basis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    **kwargs
        Additional keyword arguments passed to embedding.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    return embedding(adata, basis="umap", **kwargs)


def highly_variable_genes(adata: AnnData,
                          *,
                          log: bool = True,
                          shared_axes: bool = False,
                          opacity: float = 0.25,
                          width: int | None = None,
                          height: int | None = None,
                          return_fig: bool = False,
                          ):
    """
    Create scatter plots comparing normalized and non-normalized variances of genes
    against their mean expression, highlighting highly variable genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain 'means', 'variances', 'variances_norm',
        and 'highly_variable' columns in adata.var.
    log : bool
        If True, use logarithmic scale for both axes.
    shared_axes : bool
        If True, share the same scale across both plots.
    opacity : float
        Opacity of markers in scatter plots (0 to 1).
    width : int | None
        Width of the figure in pixels.
    height : int | None
        Height of the figure in pixels.
    return_fig : bool
        If True, return the figure instead of displaying it.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """

    fig = make_subplots(rows=1, cols=2,
                        shared_xaxes=shared_axes, shared_yaxes=shared_axes)
    norm_var_px = px.scatter(
        adata.var, x="means", y="variances_norm", color="highly_variable",
        hover_name=adata.var_names, opacity=opacity,
        category_orders={"highly_variable": [True, False]},
        color_discrete_sequence=[pio.templates[pio.templates.default].layout.colorway[1],
                                 pio.templates[pio.templates.default].layout.colorway[0]])
    var_px = px.scatter(
        adata.var, x="means", y="variances", color="highly_variable",
        hover_name=adata.var_names, opacity=opacity,
        category_orders={"highly_variable": [True, False]},
        color_discrete_sequence=[pio.templates[pio.templates.default].layout.colorway[1],
                                 pio.templates[pio.templates.default].layout.colorway[0]])
    for trace in norm_var_px["data"]:
        fig.add_trace(trace, row=1, col=1)
        fig.update_traces(showlegend=False)
    for trace in var_px["data"]:
        fig.add_trace(trace, row=1, col=2)

    if log:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        fig.update_yaxes(title="variance of genes (not normalized)")
        fig.update_xaxes(title="mean expression of genes")
    fig.update_yaxes(title="variance (normalized)", row=1, col=1)
    fig.update_yaxes(title="variance (not normalized)", row=2, col=1)
    fig.update_xaxes(title="mean expression")
    fig.for_each_trace(lambda t: t.update(
        name={"False": "other", "True": "highly variable"}[t.name]))
    fig.update_layout(width=width, height=height, legend_title_text="",
                      title="Highly variable genes")
    if return_fig:
        return fig
    fig.show()
    return None


def pca_variance_ratio(adata: AnnData,
                       n_pcs: int | None = None,
                       *, return_fig: bool = False):
    """
    Create a scree plot showing explained variance ratio for principal components.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain PCA results in adata.uns['pca']
        with 'variance_ratio' key.
    n_pcs : int | None
        Number of principal components to plot. If None, all available
        components will be shown.
    return_fig : bool
        If True, return the figure instead of displaying it.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    y = adata.uns["pca"]["variance_ratio"][:n_pcs]
    x = np.arange(1, len(y)+1)
    plot_df = pd.DataFrame({"ranking": x, "explained variance": y,
                            "PC": [f"PC{i}" for i in x]}).set_index("ranking")
    fig = px.line(plot_df, y="explained variance", markers=True, hover_name="PC")
    for c, val in enumerate(x):
        fig.add_annotation(
            x=val,
            y=y[c],
            text=f"PC{val}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            textangle=90)
    fig.update_layout(yaxis_tickformat="%", title=" PCA scree plot")
    if return_fig:
        return fig
    fig.show()
    return None


def save_fig(fig: go.Figure,
             savepath: str | Path,
             *,
             dragmode: str = "pan",
             margin: dict | None = None,
             config: dict | None = None,
             width: int | None = None,
             height: int | None = None,
             save_html: bool = True,
             save_png: bool = True,
             **kwargs) -> None:
    """
    Plot saving function with adjusted defaults.

    Parameters
    ----------
    fig : go.Figure
        Figure to save.
    savepath : str | Path
        Path where the figure should be saved. File extension will be added automatically
        based on the output format(s).
    dragmode : str
        Plotly dragmode setting for the figure (e.g., 'pan', 'zoom', 'select').
    margin : dict | None
        Dictionary specifying plot margins. If None, defaults to
        {"l": 30, "r": 30, "t": 30, "b": 30}.
    config : dict | None
        Plotly config dictionary. If None, defaults to
        {"scrollZoom": True, "displaylogo": False}.
    width: int | None
        Figure width in px.
    height: int | None
        Figure height in px.
    save_html : bool
        If True, save the figure as an interactive HTML file.
    save_png : bool
        If True, save the figure as a static PNG file.
    **kwargs
        Additional keyword arguments passed to fig.write_html().

    Returns
    -------
    None
        Function saves the figure to disk and returns nothing.
    """
    savepath = Path(savepath)
    if not margin:
        margin = {"l": 30, "r": 30, "t": 30, "b": 30}
    if not config:
        config = {"scrollZoom": True, "displaylogo": False}

    fig.update_layout(dragmode=dragmode, margin=margin)
    if save_html:
        fig.write_html(savepath.with_suffix(".html"),
                       config=config,
                       **kwargs)
    if save_png:
        fig.write_image(savepath.with_suffix(".png"))
