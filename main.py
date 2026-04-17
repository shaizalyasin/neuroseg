import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loading import load_data, summarise
from src.data_preprocessing import preprocess
from src.segment.correlation import compute_correlation_map, segment_correlation
from src.segment.cellpose_seg import segment_cellpose
from src.traces import extract_traces
from src.visualize import plot_segmentation_overlay, plot_traces, plot_comparison


def main():
    """End-to-end pipeline for neuron segmentation in calcium imaging data.

    Usage:
        python main.py --data data/6s.tif
        python main.py --data data/Medien1.avi
    """
    parser = argparse.ArgumentParser(description="Neuron segmentation pipeline")
    parser.add_argument("--data", required=True, help="Path to input file")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save output figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load
    print("=" * 50)
    print("STEP 1: Loading data")
    print("=" * 50)
    stack = load_data(args.data)
    summarise(stack, args.data)

    if stack.ndim != 3:
        print("ERROR: Need a temporal stack (T, H, W), got a 2D image.")
        return

    # Step 2: Preprocess
    print("\n" + "=" * 50)
    print("STEP 2: Preprocessing")
    print("=" * 50)
    dff = preprocess(stack)
    summarise(dff, "Preprocessed dF/F")

    # Projections for visualization and Cellpose input
    mean_proj_raw = np.mean(stack, axis=0)   # raw projection for Cellpose
    mean_proj_dff = np.mean(dff, axis=0)     # dF/F projection for display

    # Step 3a: Correlation segmentation
    print("\n" + "=" * 50)
    print("STEP 3a: Correlation map segmentation")
    print("=" * 50)
    corr_map = compute_correlation_map(dff)
    labels_corr = segment_correlation(corr_map)

    # Step 3b: Cellpose segmentation
    print("\n" + "=" * 50)
    print("STEP 3b: Cellpose segmentation")
    print("=" * 50)
    labels_cellpose = segment_cellpose(mean_proj_raw)

    # Step 4: Extract traces
    print("\n" + "=" * 50)
    print("STEP 4: Extracting traces")
    print("=" * 50)

    print("Correlation ROIs:")
    traces_corr = extract_traces(dff, labels_corr)
    print("Cellpose ROIs:")
    traces_cellpose = extract_traces(dff, labels_cellpose)

    # Step 5: Visualize
    print("\n" + "=" * 50)
    print("STEP 5: Saving visualizations")
    print("=" * 50)

    # Segmentation overlays
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_segmentation_overlay(mean_proj_dff, labels_corr,
                              title="Correlation method", ax=ax,
                              save_path=f"{args.output_dir}/correlation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_segmentation_overlay(mean_proj_raw, labels_cellpose,
                              title="Cellpose method", ax=ax,
                              save_path=f"{args.output_dir}/cellpose.png")
    plt.close(fig)

    # Activity traces
    fig = plot_traces(traces_corr, n=10,
                      title="Correlation",
                      save_path=f"{args.output_dir}/traces_correlation.png")
    plt.close(fig)

    fig = plot_traces(traces_cellpose, n=10,
                      title="Cellpose",
                      save_path=f"{args.output_dir}/traces_cellpose.png")
    plt.close(fig)

    # Comparison
    fig = plot_comparison(mean_proj_dff, labels_corr, labels_cellpose,
                          traces_corr, traces_cellpose,
                          save_path=f"{args.output_dir}/comparison.png")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"  Correlation: {labels_corr.max()} neurons")
    print(f"  Cellpose:    {labels_cellpose.max()} neurons")


if __name__ == "__main__":
    main()
