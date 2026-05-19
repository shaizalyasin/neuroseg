import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loading import load_data, summarise
from src.data_preprocessing import preprocess
from src.segment.correlation import compute_correlation_map, segment_correlation
from src.segment.cellpose import segment_cellpose
from src.traces import extract_traces, compute_snr
from src.evaluate import match_rois, global_iou
from src.visualize import plot_segmentation_overlay, plot_traces, plot_comparison
from src.logger import get_next_run_id, log_experiment, save_config


def main():
    parser = argparse.ArgumentParser(description="Neuron segmentation pipeline")
    parser.add_argument("--data", required=True, help="Path to input file")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--indicator", default="cytoplasmic",
                        choices=["cytoplasmic", "nls"])
    parser.add_argument("--method", default="both",
                        choices=["both", "correlation", "cellpose"])
    parser.add_argument("--temporal-bin", type=int, default=None)
    parser.add_argument("--no-motion-correct", action="store_true")
    parser.add_argument("--dff-window", type=int, default=30)
    parser.add_argument("--cp-model", default="cyto3")
    parser.add_argument("--cp-diameter", type=float, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_corr = args.method in ("both", "correlation")
    run_cp = args.method in ("both", "cellpose")

    config = {
        "dataset": os.path.basename(args.data),
        "data_path": args.data,
        "indicator": args.indicator,
        "method": args.method,
        "motion_correct": not args.no_motion_correct,
        "temporal_bin": args.temporal_bin,
        "dff_window": args.dff_window,
        "cp_model": args.cp_model,
        "cp_diameter": args.cp_diameter,
        "notes": args.notes,
    }

    run_id = get_next_run_id()
    print(f"\n{'='*50}\n  EXPERIMENT RUN {run_id:03d}\n{'='*50}\n")

    # Step 1: Load
    print("STEP 1: Loading data")
    stack = load_data(args.data)
    summarise(stack, args.data)
    if stack.ndim != 3:
        print("ERROR: Need (T, H, W) stack")
        return

    # Step 2: Preprocess (no background subtraction, fixed dF/F)
    print("\nSTEP 2: Preprocessing")
    dff = preprocess(
        stack,
        do_motion_correct=not args.no_motion_correct,
        temporal_bin_size=args.temporal_bin,
        dff_window=args.dff_window,
        edge_crop=5,
    )
    summarise(dff, "Preprocessed dF/F")

    # Projections
    max_proj_raw = np.max(stack, axis=0)   # for Cellpose
    mean_proj_dff = np.mean(dff, axis=0)   # for visualisation

    # Step 3a: Correlation
    labels_corr = None
    traces_corr = None
    snr_corr = np.array([])
    if run_corr:
        print("\nSTEP 3a: Correlation segmentation")
        corr_map = compute_correlation_map(dff)
        labels_corr = segment_correlation(corr_map)
        if labels_corr.max() > 0:
            traces_corr = extract_traces(dff, labels_corr)
            snr_corr = compute_snr(traces_corr)

    # Step 3b: Cellpose
    labels_cell = None
    traces_cell = None
    snr_cell = np.array([])
    if run_cp:
        print("\nSTEP 3b: Cellpose segmentation")
        labels_cell = segment_cellpose(
            max_proj_raw,
            model_type=args.cp_model,
            diameter=args.cp_diameter,
            use_gpu=False,
        )
        if labels_cell.max() > 0:
            traces_cell = extract_traces(dff, labels_cell)
            snr_cell = compute_snr(traces_cell)

    # Step 4: Cross‑method comparison
    global_iou_val = None
    n_matched = None
    if run_corr and run_cp and labels_corr is not None and labels_cell is not None:
        if labels_corr.max() > 0 and labels_cell.max() > 0:
            print("\nSTEP 4: Cross‑method evaluation")
            matched, only_corr, only_cell = match_rois(labels_corr, labels_cell)
            n_matched = len(matched)
            global_iou_val = global_iou(labels_corr, labels_cell)
            print(f"  Global IoU: {global_iou_val:.3f}")
            print(f"  Matched ROIs: {n_matched}")

    # Step 5: Logging
    print("\nSTEP 5: Logging experiment")
    if run_corr and labels_corr is not None:
        log_experiment(run_id, config, "correlation",
                       labels_corr.max(), snr_corr,
                       global_iou_val, n_matched)
    if run_cp and labels_cell is not None:
        log_experiment(run_id, config, "cellpose",
                       labels_cell.max(), snr_cell,
                       global_iou_val, n_matched)
    save_config(run_id, config, args.output_dir)

    # Step 6: Visualisation
    print("\nSTEP 6: Saving visualisations")
    if run_corr and labels_corr is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_segmentation_overlay(mean_proj_dff, labels_corr,
                                  title="Correlation method", ax=ax,
                                  save_path=f"{args.output_dir}/correlation.png")
        plt.close(fig)
        if traces_corr is not None and traces_corr.shape[0] > 0:
            fig = plot_traces(traces_corr, n=10,
                              title="Correlation",
                              save_path=f"{args.output_dir}/traces_correlation.png")
            plt.close(fig)

    if run_cp and labels_cell is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_segmentation_overlay(max_proj_raw, labels_cell,
                                  title="Cellpose method", ax=ax,
                                  save_path=f"{args.output_dir}/cellpose.png")
        plt.close(fig)
        if traces_cell is not None and traces_cell.shape[0] > 0:
            fig = plot_traces(traces_cell, n=10,
                              title="Cellpose",
                              save_path=f"{args.output_dir}/traces_cellpose.png")
            plt.close(fig)

    if run_corr and run_cp and labels_corr is not None and labels_cell is not None:
        fig = plot_comparison(mean_proj_dff, labels_corr, labels_cell,
                              traces_corr, traces_cell,
                              save_path=f"{args.output_dir}/comparison.png")
        plt.close(fig)

    print(f"\nDONE — Run {run_id:03d}")
    if run_corr and labels_corr is not None:
        print(f"  Correlation: {labels_corr.max()} neurons, SNR = {np.nanmean(snr_corr):.2f}")
    if run_cp and labels_cell is not None:
        print(f"  Cellpose:    {labels_cell.max()} neurons, SNR = {np.nanmean(snr_cell):.2f}")
    print(f"  Results: {args.output_dir}/")
    print(f"  CSV log: experiments.csv")


if __name__ == "__main__":
    main()
