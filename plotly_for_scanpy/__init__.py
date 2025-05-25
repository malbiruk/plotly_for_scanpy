"""
Plotly-based visualization tools for Scanpy data.

This package provides interactive Plotly alternatives to scanpy.pl visualization functions.
"""

from .dynamic_downsampling import dd
from .pl import (
    dotplot,
    embedding,
    highly_variable_genes,
    pca,
    pca_variance_ratio,
    qc_metrics,
    savefig,
    umap,
    volcano,
)

# Make available via the main namespace
__all__ = [
    "dd",
    "embedding",
    "umap",
    "pca",
    "qc_metrics",
    "highly_variable_genes",
    "dotplot",
    "volcano",
    "savefig",
    "pca_variance_ratio",
]
