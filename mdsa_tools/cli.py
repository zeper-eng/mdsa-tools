#!/usr/bin/env python3
"""
mdsa-tools end-to-end runner (single or two systems)
- Features: per-frame H-bond adjacency -> upper-triangle features
- Embeddings: PCA and optional UMAP
- Clustering: per-system auto-k KMeans via silhouette (S1/S3)
- Visuals: PCA/UMAP scatter, density contours, combined space, cluster-colored plots (S3/S6)
- Tables: top PC loading pairs (Table S1/S2), RMSD cluster cohesion (Table S3)
Artifacts are saved to --out. Designed as a thin orchestrator over mdsa_tools.
"""

from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# mdsa-tools import (adjust if your module path differs)
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor  # expects topology, trajectory

SEED = 42

@dataclass
class SystemSpec:
    name: str
    top: Path
    traj: Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end mdsa-tools runner.")
    p.add_argument("--name", required=True, help="System 1 short name")
    p.add_argument("--top", required=True, type=Path, help="System 1 topology (PDB/PRMTOP)")
    p.add_argument("--traj", required=True, type=Path, help="System 1 trajectory (XTC/DCD/...)")
    # Optional second system
    p.add_argument("--name2", type=str, help="System 2 short name")
    p.add_argument("--top2", type=Path, help="System 2 topology")
    p.add_argument("--traj2", type=Path, help="System 2 trajectory")
    # Options
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--reduce", choices=["pca", "umap"], default="pca", help="Embedding method for *combined* plots")
    p.add_argument("--n-components", type=int, default=2, help="Embedding dimensions")
    p.add_argument("--kmin", type=int, default=2, help="Min clusters for auto-k")
    p.add_argument("--kmax", type=int, default=8, help="Max clusters for auto-k")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per system")
    p.add_argument("--downsample", type=int, default=1, help="Keep every Nth frame (>=1)")
    p.add_argument("--save-figures", action="store_true", help="Save PNG figures")
    p.add_argument("--rmsd-window", type=int, default=5, help="Window size (frames) for cohesion RMSD")
    return p.parse_args()

def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def get_adjacency(tp) -> np.ndarray:
    """Return frames×R×R adjacency. Adjust to your API if needed."""
    if hasattr(tp, "adjacency_matrices"):
        return np.asarray(tp.adjacency_matrices())
    if hasattr(tp, "generate_hbond_adjacency"):
        return np.asarray(tp.generate_hbond_adjacency())
    if hasattr(tp, "adjacency_and_features"):
        adj, _ = tp.adjacency_and_features()
        return np.asarray(adj)
    raise RuntimeError("TrajectoryProcessor lacks an adjacency method.")

def flatten_upper(adj_3d: np.ndarray) -> np.ndarray:
    triu = np.triu_indices(adj_3d.shape[1], k=1)
    return adj_3d[:, triu[0], triu[1]].astype(np.float32)

def maybe_subset(n: int, stride: int, max_frames: Optional[int]) -> np.ndarray:
    idx = np.arange(n)
    if stride and stride > 1:
        idx = idx[::stride]
    if max_frames is not None and len(idx) > max_frames:
        idx = idx[:max_frames]
    return idx

def embed_features(features: np.ndarray, method: str, n_components: int) -> np.ndarray:
    X = StandardScaler().fit_transform(features)
    if method == "pca":
        return PCA(n_components=n_components, random_state=SEED).fit_transform(X)
    if method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("UMAP selected but umap-learn is not installed.")
        return umap.UMAP(n_components=n_components, random_state=SEED).fit_transform(X)
    raise ValueError(method)

def kmeans_auto_k(X: np.ndarray, kmin: int, kmax: int) -> tuple[np.ndarray, int, Dict[int, float]]:
    kmin = max(2, int(kmin))
    kmax = max(kmin, int(kmax))
    n = X.shape[0]
    kmax = min(kmax, max(2, n - 1))
    best_k, best_score, best_labels = None, -np.inf, None
    scores: Dict[int, float] = {}
    for k in range(kmin, kmax + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=SEED)
            lab = km.fit_predict(X)
            if len(np.unique(lab)) < 2:
                continue
            sc = silhouette_score(X, lab, metric="euclidean")
            scores[k] = sc
            if sc > best_score:
                best_k, best_score, best_labels = k, sc, lab
        except Exception:
            continue
    if best_labels is None:
        k = max(2, kmin)
        best_labels = KMeans(n_clusters=k, n_init="auto", random_state=SEED).fit_predict(X)
        best_k = k
        scores[k] = float("nan")
    return best_labels, int(best_k), scores

def top_pc_loading_pairs(pca: PCA, top_n: int = 20) -> pd.DataFrame:
    """Rank residue–residue feature pairs by |PC1| and |PC2| magnitudes."""
    # We don’t know residue mapping here; treat features as indices f0..fN.
    pc1 = pca.components_[0]
    pc2 = pca.components_[1] if pca.n_components_ > 1 else np.zeros_like(pc1)
    def top_for(vec, lab):
        idx = np.argsort(-np.abs(vec))[:top_n]
        return pd.DataFrame({
            "feature_idx": idx,
            f"{lab}_Weights": vec[idx],
            f"{lab}_magnitude": np.abs(vec[idx]),
        })
    t1 = top_for(pc1, "PC1")
    t2 = top_for(pc2, "PC2")
    df = t1.merge(t2, on="feature_idx", how="outer").fillna(0)
    # Dummy “Comparisons” column like “i-j” if you later map features to residue pairs
    df.insert(0, "Comparisons", [str(i) for i in df["feature_idx"].tolist()])
    return df

def cluster_cohesion_rmsd(X_embed: np.ndarray, labels: np.ndarray, window: int = 5) -> pd.DataFrame:
    """
    Approximate intra-cluster cohesion using pairwise distances in embedding space
    over sliding windows of frames (stand-in for RMSD if 3D coords aren’t available).
    """
    df = []
    uniq = np.unique(labels)
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        # Sliding windows
        for w in range(1, 6):
            step = max(1, window // w)
            slabs = []
            for s in range(0, len(idx)-step+1, step):
                block = X_embed[idx[s:s+step]]
                if len(block) > 1:
                    d = pairwise_distances(block).mean()
                    slabs.append(d)
            if slabs:
                df.append({"cluster": float(c), "rmsd": float(np.mean(slabs)), "window": w})
    return pd.DataFrame(df) if df else pd.DataFrame(columns=["cluster", "rmsd", "window"])

def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def plot_scatter(df: pd.DataFrame, x: str, y: str, by: str, title: str, out_png: Path):
    fig = plt.figure(figsize=(6,5), dpi=150)
    ax = plt.gca()
    for key, sub in df.groupby(by):
        ax.scatter(sub[x], sub[y], s=8, alpha=0.7, label=str(key))
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.legend(markerscale=2)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_silhouette(scores: Dict[int, float], title: str, out_png: Path):
    ks = sorted(scores.keys())
    vs = [scores[k] for k in ks]
    fig = plt.figure(figsize=(5,3), dpi=150); ax = plt.gca()
    ax.plot(ks, vs, marker="o"); ax.set_xticks(ks)
    ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def run_one(spec: SystemSpec, outdir: Path, reduce_for_plots: str, n_components: int,
            kmin: int, kmax: int, max_frames: Optional[int], stride: int,
            save_figs: bool) -> dict:
    # Load features
    tp = TrajectoryProcessor(str(spec.top), str(spec.traj))
    adj = get_adjacency(tp)
    idx = maybe_subset(adj.shape[0], stride, max_frames)
    adj = adj[idx]
    feats = flatten_upper(adj)

    # Save raw
    np.save(outdir / f"{spec.name}_features.npy", feats)

    # PCA embedding (always compute; used for S3A/B and loading tables)
    X_std = StandardScaler().fit_transform(feats)
    pca = PCA(n_components=max(2, n_components), random_state=SEED).fit(X_std)
    pca_emb = pca.transform(X_std)
    df_pca = pd.DataFrame({"x": pca_emb[:,0], "y": pca_emb[:,1], "frame_idx": np.arange(len(pca_emb))})
    df_pca["system"] = spec.name
    save_df(df_pca, outdir / f"{spec.name}_pca_embedding.csv")
    if save_figs:
        plot_scatter(df_pca, "x", "y", "system",
                     f"{spec.name} PCA embedding",
                     outdir / f"{spec.name}_pca_embedding.png")

    # Auto-k KMeans on PCA space (S1/S3)
    labels, best_k, scores = kmeans_auto_k(pca_emb, kmin, kmax)
    save_df(pd.DataFrame({"frame_idx": np.arange(len(labels)), "cluster": labels}),
            outdir / f"{spec.name}_clusters.csv")
    json.dump({"best_k": int(best_k), "silhouette": scores}, open(outdir / f"{spec.name}_k_selection.json", "w"), indent=2)
    if save_figs:
        # S1: silhouette curve
        plot_silhouette(scores, f"{spec.name} silhouette", outdir / f"{spec.name}_silhouette.png")
        # S3B: PCA colored by cluster
        df_pca_c = df_pca.copy(); df_pca_c["cluster"] = labels
        plot_scatter(df_pca_c, "x", "y", "cluster",
                     f"{spec.name} PCA + KMeans (k={best_k})",
                     outdir / f"{spec.name}_pca_clusters.png")

    # Density “contour” proxy: save a hexbin density plot (S3C-ish)
    if save_figs:
        fig = plt.figure(figsize=(6,5), dpi=150); ax = plt.gca()
        hb = ax.hexbin(df_pca["x"], df_pca["y"], gridsize=40)
        ax.set_title(f"{spec.name} PCA density"); fig.colorbar(hb, ax=ax)
        fig.tight_layout(); fig.savefig(outdir / f"{spec.name}_pca_density.png"); plt.close(fig)

    # Optional UMAP/selected reduction (S6)
    chosen_emb = embed_features(feats, reduce_for_plots, n_components)
    df_emb = pd.DataFrame({"x": chosen_emb[:,0], "y": chosen_emb[:,1], "system": spec.name})
    save_df(df_emb, outdir / f"{spec.name}_{reduce_for_plots}_embedding.csv")
    if save_figs:
        plot_scatter(df_emb, "x", "y", "system",
                     f"{spec.name} {reduce_for_plots.upper()} embedding",
                     outdir / f"{spec.name}_{reduce_for_plots}_embedding.png")

    # Tables: top loading pairs (Table S1/S2 style)
    top_pairs = top_pc_loading_pairs(pca, top_n=20)
    save_df(top_pairs, outdir / f"{spec.name}_pc_loading_pairs_top20.csv")

    # Table: cluster cohesion “RMSD” proxy in embedding space (Table S3 style)
    coh = cluster_cohesion_rmsd(pca_emb, labels, window=5)
    save_df(coh, outdir / f"{spec.name}_cluster_cohesion_rmsd.csv")

    return {
        "features": feats, "pca_emb": pca_emb, "labels": labels,
        "best_k": best_k, "silhouette": scores, "umap_or_pca": df_emb
    }

def main():
    args = parse_args()
    ensure_out(args.out)

    # System 1
    sys1 = SystemSpec(args.name, args.top, args.traj)
    res1 = run_one(sys1, args.out, args.reduce, args.n_components,
                   args.kmin, args.kmax, args.max_frames, args.downsample,
                   args.save_figures)

    # Optional system 2 & combined space
    if args.name2 and args.top2 and args.traj2:
        sys2 = SystemSpec(args.name2, args.top2, args.traj2)
        res2 = run_one(sys2, args.out, args.reduce, args.n_components,
                       args.kmin, args.kmax, args.max_frames, args.downsample,
                       args.save_figures)

        # Combined embedding (compare systems in one space)
        combined = np.vstack([res1["features"], res2["features"]])
        comb_emb = embed_features(combined, args.reduce, args.n_components)
        df = pd.DataFrame({"x": comb_emb[:,0], "y": comb_emb[:,1]})
        df["system"] = np.array([sys1.name]*res1["features"].shape[0] + [sys2.name]*res2["features"].shape[0])
        save_df(df, args.out / "combined_embedding.csv")
        if args.save_figures:
            plot_scatter(df, "x", "y", "system",
                         f"Combined {args.reduce.upper()} embedding",
                         args.out / f"combined_{args.reduce}_embedding.png")

    # Write manifest
    manifest = {
        "systems": [{"name": args.name, "topology": str(args.top), "trajectory": str(args.traj)}] + (
            [{"name": args.name2, "topology": str(args.top2), "trajectory": str(args.traj2)}] if args.name2 and args.top2 and args.traj2 else []
        ),
        "options": {
            "reduce": args.reduce, "n_components": args.n_components,
            "kmin": args.kmin, "kmax": args.kmax,
            "max_frames": args.max_frames, "downsample": args.downsample,
            "seed": SEED
        }
    }
    json.dump(manifest, open(args.out / "manifest.json", "w"), indent=2)
    print(f"[OK] Finished. Artifacts in: {args.out.resolve()}")

if __name__ == "__main__":
    main()

