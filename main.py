import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loading import load_data, summarise, extract_fps
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
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate in Hz (required if not found in file metadata)")
    parser.add_argument("--method", default="both",
                        choices=["both", "correlation", "cellpose"])
    parser.add_argument("--temporal-bin", type=int, default=None)
    parser.add_argument("--no-motion-correct", action="store_true")
    parser.add_argument("--dff-window-sec", type=float, default=30.0,
                        help="dF/F baseline window in seconds (default: 30)")
    parser.add_argument("--cp-model", default="cyto3")
    parser.add_argument("--cp-diameter", type=float, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.indicator == "nls" and args.cp_model == "cyto3":
        args.cp_model = "nuclei"

    run_corr = args.method in ("both", "correlation")
    run_cp = args.method in ("both", "cellpose")

    config = {
        "dataset": os.path.basename(args.data),
        "data_path": args.data,
        "indicator": args.indicator,
        "fps": args.fps,
        "method": args.method,
        "motion_correct": not args.no_motion_correct,
        "temporal_bin": args.temporal_bin,
        "dff_window_sec": args.dff_window_sec,
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

    auto_fps = extract_fps(args.data)
    if args.fps is None:
        if auto_fps is not None:
            args.fps = auto_fps
            print(f"  Auto-detected FPS from file metadata: {args.fps:.2f} Hz")
        else:
            args.fps = 6.0
    else:
        print(f"  Using provided FPS: {args.fps:.2f} Hz")
        if auto_fps is not None and abs(args.fps - auto_fps) > 0.1:
            print(f"  Warning: Provided FPS ({args.fps}) differs from file metadata ({auto_fps:.2f})")

    # Update config with final fps
    config["fps"] = args.fps

    # Step 2: Preprocess
    print("\nSTEP 2: Preprocessing")
    dff = preprocess(
        stack,
        fps=args.fps,
        do_motion_correct=not args.no_motion_correct,
        temporal_bin_size=args.temporal_bin,
        dff_window_sec=args.dff_window_sec,
        edge_crop=5,
    )
    summarise(dff, "Preprocessed dF/F")

    # Projections
    mean_proj_raw = np.mean(stack, axis=0)
    max_proj_raw  = np.max(stack, axis=0)

    if args.indicator == "nls":
        proj_for_cellpose = mean_proj_raw    
        print("  Using MEAN projection for Cellpose (NLS indicator)")
    else:
        proj_for_cellpose = max_proj_raw     
        print("  Using MAX projection for Cellpose (cytoplasmic indicator)")

    # Step 3a: Correlation
    labels_corr = None
    traces_corr = None
    snr_corr = np.array([])
    if run_corr:
        print("\nSTEP 3a: Correlation segmentation")
        corr_map = compute_correlation_map(dff)
        labels_corr = segment_correlation(corr_map, indicator=args.indicator)
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
            proj_for_cellpose,
            model_type=args.cp_model,
            diameter=args.cp_diameter,
            indicator=args.indicator,
            use_gpu=False,
        )
        if labels_cell.max() > 0:
            traces_cell = extract_traces(dff, labels_cell)
            snr_cell = compute_snr(traces_cell)

    # Step 4: Cross-method comparison
    global_iou_val = None
    n_matched = None
    if run_corr and run_cp and labels_corr is not None and labels_cell is not None:
        if labels_corr.max() > 0 and labels_cell.max() > 0:
            print("\nSTEP 4: Cross-method evaluation")
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
    overlay_image = mean_proj_raw

    print("\nSTEP 6: Saving visualisations")
    if run_corr and labels_corr is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_segmentation_overlay(overlay_image, labels_corr,
                                  title="Correlation method", ax=ax)
        fig.savefig(f"{args.output_dir}/correlation.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {args.output_dir}/correlation.png")
        plt.close(fig)
        if traces_corr is not None and traces_corr.shape[0] > 0:
            fig = plot_traces(traces_corr, n=10,
                              title="Correlation",
                              save_path=f"{args.output_dir}/traces_correlation.png")
            plt.close(fig)

    if run_cp and labels_cell is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_segmentation_overlay(overlay_image, labels_cell,
                                  title="Cellpose method", ax=ax)
        fig.savefig(f"{args.output_dir}/cellpose.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {args.output_dir}/cellpose.png")
        plt.close(fig)
        if traces_cell is not None and traces_cell.shape[0] > 0:
            fig = plot_traces(traces_cell, n=10,
                              title="Cellpose",
                              save_path=f"{args.output_dir}/traces_cellpose.png")
            plt.close(fig)

    if run_corr and run_cp and labels_corr is not None and labels_cell is not None:
        fig = plot_comparison(overlay_image, labels_corr, labels_cell,
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
