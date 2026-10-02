# Breast Cancer pCR Prediction System

A web-based radiomics tool for predicting pathological complete response (pCR) to neoadjuvant chemotherapy (NAC) in breast cancer from MRI.

## Overview

This repository implements the full analysis pipeline of our study:

1. **Compactness** (IBSI Compactness 1) — a tumor morphological feature computed from PyRadiomics shape features:
   `Compactness = V / (sqrt(pi) * A^1.5)`
2. **3D Topological Intratumoral Heterogeneity (ITH)** — computed from PyRadiomics voxel-based texture feature maps + KMeans (K=4) clustering + 3D connected-component topology analysis.

## Prediction models

- **MSH model** (imaging only):
  `MSH score = 1 / (1 + exp(-(-1.0096 + 61.8292*Compactness - 2.4438*3D_Topo_ITH)))` — cutoff 0.442

- **Fusion model** (imaging + clinicopathological):
  `P(pCR) = 1 / (1 + exp(-(-1.002260 + 0.002288*stage - 1.037019*hr + 1.362167*her2 - 2.495751*topo_ith + 63.488501*compactness)))` — cutoff 0.391

## Directory structure

```
├── backend/            # FastAPI server + feature extraction + visualization
│   ├── main.py         # API endpoints & prediction formulas
│   ├── features.py     # Compactness & 3D Topological ITH computation
│   ├── visual.py       # dashboard rendering
│   └── requirements.txt
├── frontend/           # web UI (HTML / CSS / JS)
└── Dockerfile          # containerized deployment
```

## Quick start (Docker, recommended)

```bash
docker build -t breast-pcr .
docker run -p 8000:8000 breast-pcr
# open http://localhost:8000
```

## Manual setup

```bash
cd backend
pip install -r requirements.txt
pip install "pyradiomics==3.0.1" --no-build-isolation
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Usage

1. Upload the original MRI (`Image.nii.gz`) and the delineated ROI (`ROI.nii.gz`).
2. Select mT stage, HR status and HER2 status.
3. The tool computes both radiomic features, outputs pCR probabilities from both models, and visualizes tumor morphology and intratumoral habitat heterogeneity.

## Citation

If you use this code in your work, please cite our paper.
