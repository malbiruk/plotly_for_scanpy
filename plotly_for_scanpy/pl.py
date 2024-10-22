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
    width: int | None = None,
    height: int | None = None,
    layout: str = "vertical",
    *,
    template: str | None = None,
    mt_col: str = "mt",
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
    layout_err_msg = 'layout should be "vertical" or "horizontal"'
    if template is None:
        template = pio.templates.default

    if layout not in ["vertical", "horizontal"]:
        raise ValueError(layout_err_msg)

    if not width:
        width = 1200 if layout == "horizontal" else None
    if not height:
        height = None if layout == "horizontal" else 1000

    counts_per_cell = adata.X.sum(axis=1).A.T[0]
    genes_per_cell = (adata.X > 0).sum(axis=1).T.A[0]
    if mt_col not in adata.var.columns:
        adata.var[mt_col] = adata.var_names.str.startswith("MT-")
    mito_counts = adata[:, adata.var[mt_col]].X.sum(axis=1).T.A[0]
    mito_pct = mito_counts / counts_per_cell * 100

    n_rows = 3 if layout == "vertical" else 1
    n_cols = 1 if layout == "vertical" else 3

    fig = make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.075, vertical_spacing=0.1,
                        subplot_titles=[
                            "N of UMI per cell",
                            "N of genes per cell",
                            "% of mitochondrial expression per cell",
                        ])
    if layout == "vertical":
        fig.add_trace(go.Histogram(x=counts_per_cell), row=1, col=1)
        fig.add_trace(go.Histogram(x=genes_per_cell), row=2, col=1)
        fig.add_trace(go.Histogram(x=mito_pct), row=3, col=1)
    else:
        fig.add_trace(go.Histogram(x=counts_per_cell), row=1, col=1)
        fig.add_trace(go.Histogram(x=genes_per_cell), row=1, col=2)
        fig.add_trace(go.Histogram(x=mito_pct), row=1, col=3)

    for row in range(1, n_rows+1):
        for col in range(1, n_cols+1):
            fig.update_yaxes(showticklabels=False, ticks="", showline=False, row=row, col=col)

    fig.update_layout(showlegend=False,
                      template=template, width=width, height=height)
    fig.update_annotations(font={"family": "Serif"})

    if return_fig:
        return fig

    fig.show()
    return None


def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    """Get array for basis from anndata. Just tries to add 'X_'."""
    if basis in adata.obsm:
        return adata.obsm[basis]
    if f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    err_msg = f"Could not find '{basis}' or 'X_{basis}' in .obsm"
    raise KeyError(err_msg)


def _prepare_dimension_dataframes(adata, basis, dimensions):
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
        dfs_for_plot.append(df_for_plot)

    return dfs_for_plot, bas


def _add_color_data(adata, dfs_for_plot, color):
    """
    Add color data to DataFrames.
    """
    if color is None:
        color = [""]
    else:
        color = [color] if isinstance(color, str) else list(color)

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


def _add_trace_to_figure(fig, trace, row, col, counter, color_col, dim_pair):
    """
    Add a trace to the figure with proper formatting.
    """
    legendgroup_name = f"{color_col}_{dim_pair}"
    fig.add_trace(trace, row=row, col=col)
    fig.update_traces(
        legendgrouptitle=dict(text=legendgroup_name),
        legendgroup=legendgroup_name,
        marker=dict(coloraxis=f"coloraxis{counter+1}"),
        row=row,
        col=col,
    )
    fig.update_layout({f"coloraxis{counter+1}": {"showscale": False}})


def _update_figure_layout(fig, template, marker_size, width, height):
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
    )
    fig.update_yaxes(showticklabels=False, zeroline=False, ticks="")
    fig.update_xaxes(showticklabels=False, zeroline=False, ticks="")
    fig.update_annotations(font={"family": "Serif"})


def embedding(adata: AnnData,
              basis: str,
              *,
              marker_size: float = 5,
              template: str | None = None,
              dimensions: tuple[int, int] | list[tuple[int, int]] = (0, 1),
              color: str | list[str] | None = None,
              maxcols: int = 3,
              horizontal_spacing: float = 0.1,
              vertical_spacing: float = 0.15,
              opacity: float = 0.25,
              width: int | None = None,
              height: int | None = None,
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
    maxcols : int
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
    return_fig : bool
        If True, return the figure instead of displaying it.
    **kwargs
        Additional keyword arguments passed to px.scatter.

    Returns
    -------
    go.Figure | None
        The figure object if return_fig is True, None otherwise.
    """
    if template is None:
        template = pio.templates.default

    # Prepare data
    if isinstance(dimensions, tuple):
        dimensions = [dimensions]
    dfs_for_plot, bas = _prepare_dimension_dataframes(adata, basis, dimensions)
    color = _add_color_data(adata, dfs_for_plot, color)

    # Calculate layout
    num_plots = len(color) * len(dimensions)
    cols = min(num_plots, maxcols)
    rows = int(np.ceil(num_plots / cols))

    # Create figure
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=color * len(dimensions),
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # Add subplots
    counter = 0
    for dim_pair, df_for_plot in zip(dimensions, dfs_for_plot):
        for color_col in color:
            x_col = f"{bas}{dim_pair[0]+1}"
            y_col = f"{bas}{dim_pair[1]+1}"

            px_fig = px.scatter(
                df_for_plot,
                x=x_col,
                y=y_col,
                template=template,
                color=color_col,
                opacity=opacity,
                category_orders={color_col: sorted(df_for_plot[color_col].unique())},
                **kwargs)

            for trace in px_fig["data"]:
                row = (counter // cols) + 1
                col = (counter % cols) + 1
                _add_trace_to_figure(fig, trace, row, col, counter, color_col, dim_pair)

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
    _update_figure_layout(fig, template, marker_size, width, height)

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
    embedding(adata, basis="umap", **kwargs)


def save_fig(fig: go.Figure,
             savepath: str | Path,
             *,
             dragmode: str = "pan",
             margin: dict | None = None,
             config: dict | None = None,
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
