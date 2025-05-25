# plotly_for_scanpy

Interactive Plotly-based visualization tools for Scanpy single-cell data analysis, providing drop-in alternatives to scanpy.pl functions with enhanced interactivity.

## Installation

```bash
pip install plotly_for_scanpy
```

## Features

- **Dynamic Downsampling**: Efficiently visualize large datasets with intelligent point reduction
- **Interactive Plots**: All visualizations are fully interactive with zoom, pan, and hover functionality
- **Scanpy Integration**: Works seamlessly with AnnData objects
- **Publication Quality**: Easily export to HTML or static formats

## Example Usage

```python
import scanpy as sc
import plotly_for_scanpy as psc

# Load data
adata = sc.datasets.pbmc3k()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Use interactive UMAP visualization
psc.umap(adata, color=['leiden'])

# Visualize with dynamic downsampling for large datasets
fig = psc.umap(adata, color='leiden', return_fig=True)
dynamic_fig = psc.dd(fig, resolution=10000)
dynamic_fig.show()
```

## Documentation

Full examples and tutorials:
- [Interactive Colab Demo](https://colab.research.google.com/drive/1pHt8m-wIVS2B7aUbjrlsNrNWV2JsEI-E)

## License

MIT License
