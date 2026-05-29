# neuroseg

Automatic detection and segmentation of active neuron somata in 2D+t calcium imaging data.

## Overview

Manual annotation of active neurons in calcium imaging videos is time consuming and subjective.  
This pipeline provides an automatic, end‑to‑end tool to segment active neurons and extract their activity traces.

## Background

Neurons fire action potentials this causes a temporary rise in intracellular calcium.  
Calcium imaging uses fluorescent indicators that brighten when calcium increases.  
Active neurons literally light up over time.  
This pipeline processes such videos to find and track those active cells.

## Features

- Supports `.tif`, `.avi`, and `.czi` files.
- Motion correction, temporal binning, and robust `dF/F` calculation.
- Two segmentation methods:
  - **Correlation** – uses temporal synchrony of neighbouring pixels.
  - **Cellpose** – deep learning based on morphology.
- Full experiment logging: CSV summary + YAML config per run.
- Evaluation metrics: neuron count, SNR, IoU.

## Installation

```bash
git clone https://github.com/shaizalyasin/neuroseg.git
cd neuroseg
pip install -r requirements.txt
```

## Usage

Run the pipeline using `main.py`. All configuration is done via command‑line arguments.

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to input file (`.tif`, `.avi`, `.czi`). | **required** |
| `--indicator` | Type of calcium indicator: `cytoplasmic` or `nls`. | `cytoplasmic` |
| `--method` | Segmentation method(s): `correlation`, `cellpose`, or `both`. | `both` |
| `--output-dir` | Folder where result images are saved. | `results` |
| `--no-motion-correct` | Skip motion correction. | `False` |
| `--cp-diameter` | Expected neuron diameter (pixels). Auto‑estimated if `None`. | `None` |

### Examples

```bash
python main.py --data path/to/your/example.tif --indicator cytoplasmic --method both
```

### Output

- **Visualisations:** Saved in `--output-dir` – segmentation overlays, activity traces.
- **Experiment log:** `experiments.csv` in the project root contains run ID, method, neuron count, mean SNR, global IoU, and more.
- **Configuration:** Each run stores a full `config.yaml` inside `experiments/exp_XXX/`.

## Project structure

```
neuroseg/
├── data/                    # your datasets
├── experiments/             # config.yaml per run
├── results/                 # output images
├── src/
│   ├── data_loading.py      # load .tif/.avi/.czi
│   ├── data_preprocessing.py # motion correction, dF/F
│   ├── evaluate.py          # IoU, ROI matching
│   ├── logger.py            # CSV + YAML logging
│   ├── traces.py            # trace extraction, SNR
│   ├── visualize.py         # plotting overlays and traces
│   └── segment/
│       ├── correlation.py   # correlation segmentation
│       └── cellpose.py      # deep learning segmentation
├── main.py
└── experiments.csv          # log of all runs
```

## References

- **Correlation method** - https://orgerlab.org/wp-content/uploads/2016/12/2016-Correlating.pdf
- **Cellpose** - https://github.com/MouseLand/cellpose 
