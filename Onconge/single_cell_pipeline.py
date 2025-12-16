"""
Pipeline utilities for running a complete single-cell RNA-seq analysis
on 10x Genomics Cell Ranger outputs. The script is intentionally
configurable and targets the dataset layout used in this repository.

Example usage
-------------
python single_cell_pipeline.py \
    --cellranger-root cellranger \
    --metadata metadata/patients_metadata.xlsx \
    --markers curated_annotation/curated_annotation_str.xlsx \
    --output-dir results

The pipeline performs the following steps:
1. Load each sample from ``cellranger/<sample_id>/filtered_feature_bc_matrix``
   using :func:`scanpy.read_10x_mtx`.
2. Merge samples into one :class:`~anndata.AnnData` object with ``sample_id``
   recorded in ``adata.obs``.
3. Optional: enrich the metadata using the provided Excel table.
4. Quality control (gene/cell filters and mitochondrial content) followed by
   normalization and highly-variable gene selection.
5. PCA, neighborhood graph computation, UMAP embedding, Leiden clustering.
6. Differential expression ranking per Leiden cluster.
7. Optional: marker-based cell type annotation from a curated Excel file.
8. Persist results to ``.h5ad`` and flat files for embeddings and marker genes.

The pipeline uses Scanpy defaults for clarity; thresholds can be adjusted via
CLI arguments to match dataset-specific requirements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import scanpy as sc


def discover_samples(cellranger_root: Path) -> List[Tuple[str, Path]]:
    """
    Locate 10x Genomics matrices inside ``cellranger_root``.

    Parameters
    ----------
    cellranger_root:
        Directory containing subfolders (one per sample) with a
        ``filtered_feature_bc_matrix`` directory.

    Returns
    -------
    list of tuples
        Each tuple is ``(sample_id, matrix_path)`` where ``matrix_path`` points
        to the ``filtered_feature_bc_matrix`` directory.
    """

    sample_dirs: List[Tuple[str, Path]] = []
    for sample_dir in cellranger_root.iterdir():
        matrix_dir = sample_dir / "filtered_feature_bc_matrix"
        if matrix_dir.is_dir():
            sample_dirs.append((sample_dir.name, matrix_dir))
    if not sample_dirs:
        raise FileNotFoundError(
            f"No samples found under {cellranger_root}. "
            "Each sample should contain a filtered_feature_bc_matrix directory."
        )
    return sample_dirs


def load_sample(matrix_dir: Path, sample_id: str) -> sc.AnnData:
    """Load a single sample from a Cell Ranger ``filtered_feature_bc_matrix``."""

    adata = sc.read_10x_mtx(matrix_dir, cache=True)
    adata.var_names_make_unique()
    adata.obs["sample_id"] = sample_id
    return adata


def load_all_samples(cellranger_root: Path) -> sc.AnnData:
    """Read and concatenate all samples under ``cellranger_root``."""

    loaded = [load_sample(matrix_dir, sample_id) for sample_id, matrix_dir in discover_samples(cellranger_root)]
    return sc.concat(loaded, join="outer", label="batch", keys=[s for s, _ in discover_samples(cellranger_root)])


def integrate_metadata(adata: sc.AnnData, metadata_path: Optional[Path]) -> sc.AnnData:
    """
    Merge additional sample-level metadata into ``adata.obs``.

    The metadata Excel file is expected to contain a column named ``sample_id``
    (case-insensitive). Additional columns are copied to ``adata.obs`` for the
    matching samples. Rows without a matching ``sample_id`` are ignored.
    """

    if metadata_path is None:
        return adata

    metadata = pd.read_excel(metadata_path)
    normalized_cols = {col.lower(): col for col in metadata.columns}
    if "sample_id" not in normalized_cols:
        raise KeyError(
            "Metadata file must include a 'sample_id' column to merge with obs."
        )

    sample_col = normalized_cols["sample_id"]
    metadata = metadata.set_index(sample_col)
    metadata.index = metadata.index.astype(str)

    for column in metadata.columns:
        adata.obs[column] = adata.obs["sample_id"].map(metadata[column])
    return adata


def compute_qc_metrics(adata: sc.AnnData, mt_prefix: str = "MT-") -> None:
    """Add standard QC metrics to ``adata.obs``."""

    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)


def filter_cells_and_genes(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mt_fraction: float = 0.2,
) -> sc.AnnData:
    """Apply QC filters and return a filtered AnnData copy."""

    compute_qc_metrics(adata)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs["pct_counts_mt"] <= max_mt_fraction * 100].copy()
    return adata


def basic_normalization(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    """Normalize, log-transform, and identify highly variable genes."""

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)


def run_dimensionality_reduction(
    adata: sc.AnnData,
    n_pcs: int = 50,
    neighbors_k: int = 15,
    batch_key: Optional[str] = None,
) -> None:
    """Compute PCA, neighbors, and UMAP; perform optional batch correction."""

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    if batch_key is not None and batch_key in adata.obs.columns:
        sc.pp.combat(adata, key=batch_key)

    sc.pp.neighbors(adata, n_neighbors=neighbors_k, n_pcs=n_pcs)
    sc.tl.umap(adata)


def cluster_and_rank_genes(adata: sc.AnnData, resolution: float = 0.5) -> None:
    """Leiden clustering followed by differential expression ranking."""

    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")


def load_marker_table(markers_path: Path) -> Dict[str, List[str]]:
    """
    Parse a curated marker table.

    The Excel file should include two columns: ``cell_type`` and ``genes``.
    The ``genes`` field may contain comma- or space-separated gene symbols.
    """

    df = pd.read_excel(markers_path)
    normalized_cols = {col.lower(): col for col in df.columns}
    if "cell_type" not in normalized_cols or "genes" not in normalized_cols:
        raise KeyError("Marker file must have 'cell_type' and 'genes' columns.")

    cell_col = normalized_cols["cell_type"]
    genes_col = normalized_cols["genes"]
    markers: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        genes = [gene.strip() for gene in str(row[genes_col]).replace(";", ",").replace(" ", ",").split(",") if gene.strip()]
        markers[str(row[cell_col])] = genes
    if not markers:
        raise ValueError("No marker genes parsed from the provided file.")
    return markers


def annotate_cell_types(adata: sc.AnnData, markers_path: Optional[Path]) -> None:
    """Assign cell types by scoring curated marker sets."""

    if markers_path is None:
        return

    markers = load_marker_table(markers_path)
    scores = {}
    for cell_type, genes in markers.items():
        score_key = f"score_{cell_type}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_key, use_raw=False)
        scores[cell_type] = score_key

    # Choose the marker set with the highest score for each cell
    score_cols = [col for col in adata.obs.columns if col.startswith("score_")]
    adata.obs["cell_type"] = adata.obs[score_cols].idxmax(axis=1).str.replace("score_", "")


def save_outputs(adata: sc.AnnData, output_dir: Path) -> None:
    """Persist AnnData object and useful tables to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adata.write(output_dir / "analysis.h5ad")

    embeddings = adata.obsm.get("X_umap")
    if embeddings is not None:
        embedding_df = pd.DataFrame(embeddings, index=adata.obs_names, columns=["UMAP1", "UMAP2"])
        embedding_df = pd.concat([embedding_df, adata.obs], axis=1)
        embedding_df.to_csv(output_dir / "embeddings.csv")

    if "rank_genes_groups" in adata.uns:
        sc.get.rank_genes_groups_df(adata, None).to_csv(
            output_dir / "rank_genes_groups.csv", index=False
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end single-cell RNA-seq pipeline")
    parser.add_argument("--cellranger-root", type=Path, default=Path("cellranger"), help="Root folder with Cell Ranger outputs")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional Excel file with sample-level metadata")
    parser.add_argument("--markers", type=Path, default=None, help="Optional Excel file with curated marker genes")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory to store pipeline outputs")
    parser.add_argument("--min-genes", type=int, default=200, help="Minimum genes per cell")
    parser.add_argument("--min-cells", type=int, default=3, help="Minimum cells per gene")
    parser.add_argument("--max-mt-fraction", type=float, default=0.2, help="Maximum mitochondrial fraction (0-1)")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of principal components")
    parser.add_argument("--neighbors-k", type=int, default=15, help="Neighbors for KNN graph")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden resolution")
    parser.add_argument("--batch-key", type=str, default=None, help="Batch column in obs for Combat correction")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sc.settings.set_figure_params(dpi=120)

    print("[1/7] Loading samples...")
    adata = load_all_samples(args.cellranger_root)

    print("[2/7] Integrating metadata (if provided)...")
    adata = integrate_metadata(adata, args.metadata)

    print("[3/7] Filtering low-quality cells and genes...")
    adata = filter_cells_and_genes(
        adata,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_mt_fraction=args.max_mt_fraction,
    )

    print("[4/7] Normalizing and selecting highly variable genes...")
    basic_normalization(adata)

    print("[5/7] Dimensionality reduction and neighborhood graph...")
    run_dimensionality_reduction(
        adata,
        n_pcs=args.n_pcs,
        neighbors_k=args.neighbors_k,
        batch_key=args.batch_key,
    )

    print("[6/7] Clustering and marker detection...")
    cluster_and_rank_genes(adata, resolution=args.resolution)

    print("[7/7] Optional cell type annotation...")
    annotate_cell_types(adata, args.markers)

    print("Saving outputs...")
    save_outputs(adata, args.output_dir)
    print(f"Done. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
