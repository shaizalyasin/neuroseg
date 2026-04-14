# neuroseg

Automatic segmentation of active neuron somata in 2D+t calcium imaging data.

## Overview

Calcium imaging records fluorescence over time in neural tissue — active neurons emit brighter signals as intracellular calcium rises. **neuroseg** provides an end-to-end pipeline to automatically detect and segment those active somata, and extract per-neuron activity traces.

Two segmentation strategies are implemented and compared:

- **Correlation map** — unbiased, activity-driven (Orger & Portugues, 2016)
- **Cellpose** — pretrained deep learning model (Stringer et al., 2021)

## Installation

```bash
git clone https://github.com/shaizalyasin/neuroseg.git
cd neuroseg
```

## Usage

```bash
python main.py --data --method
```

## Project Structure

```
src/
├── data_loading.py
├── data_preprocessing.py
├── segment/
│   ├── correlation.py
│   └── cellpose_seg.py
├── traces.py
└── visualize.py
```
