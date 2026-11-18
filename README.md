# Proposed AA-HUnet: Artifact-Aware Hybrid U-Net for Pneumothorax Segmentation

A robust deep learning framework for accurate pneumothorax segmentation in frontal chest radiographs under real-world clinical imaging artifacts (chest drains, ECG monitoring leads, skin folds, and surgical clips).

---

## 📁 Repository Structure

The project is organized into 5 standardized pipeline modules:

```text
Proposed AA_HUnet/
├── 01_data_pipeline/                # Data ingestion, preprocessing, caching, and split manifests
│   ├── dataset.py                   # PyTorch dataset & DICOM/PNG/JPG loaders
│   ├── cache.py                     # Fast caching utilities
│   ├── split_manifest.csv           # Train/Val/Test metadata splits
│   ├── split_info.json              # Class and split balance statistics
│   ├── dataset_audit.csv            # Original image audit
│   └── *.png                        # Distribution & lesion size plots
│
├── 02_augmentation/                 # Synthetic artifact simulation & augmentation engine
│   ├── artifacts.py                 # Injectors: chest tubes, ECG wires, skin folds, clips
│   ├── artifact_gallery.png         # Gallery of simulated artifact types
│   └── artifact_head_output.png     # Intermediate artifact attention maps
│
├── 03_model_training/               # Model architectures, loss functions, & training engine
│   ├── models/                      # AA-HybridUNet and Baseline U-Net definitions
│   ├── engine.py                    # Training and validation engine
│   ├── losses.py                    # Combined BCE + Dice + Auxiliary Artifact loss
│   ├── checkpoints/                 # Trained model weights (.pt) and configs
│   ├── runs/                        # Epoch-wise training history and logs
│   └── training_curves.png          # Convergence and loss trajectories
│
├── 04_inference_and_deployment/     # Inference engine & interactive Streamlit web application
│   ├── app.py                       # Streamlit web interface
│   ├── app_logic.py                 # DICOM/image rendering and statistics backend
│   ├── infer.py                     # Inference runner and postprocessing
│   ├── bench.py                     # Latency & throughput benchmark suite
│   ├── evalrun.py                   # Batch evaluation runner
│   ├── metrics.py                   # Evaluation metrics (Dice, IoU, Sensitivity, etc.)
│   ├── models/                      # Deployment bundles
│   ├── run_app.bat                  # 1-click launcher for the web demo
│   └── requirements.txt             # Deployment package requirements
│
├── 05_visualization/                # Evaluation results, qualitative figures, & tables
│   ├── figures/                     # Qualitative comparison & ROC/PR curves
│   ├── tables/                      # Benchmark summary tables (CSV, MD, LaTeX)
│   ├── test_results/                # Test split per-image metrics
│   ├── val_results/                 # Validation split per-image metrics
│   └── report.py                    # Report generation utilities
│
├── src/                             # Shared core Python package (aa_pneumo)
├── Complete Code.ipynb              # Master research notebook
├── requirements.txt                 # Global dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Quick Start: Streamlit Web Demo

You can launch the interactive web demo directly from either the root or `04_inference_and_deployment`:

### Method 1: Using the 1-click batch script (Windows)
Double-click `04_inference_and_deployment/run_app.bat`.

### Method 2: Command Line
```bash
# Navigate to the deployment folder
cd 04_inference_and_deployment

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🧠 Key Innovations of AA-HUnet

1. **Dual-Head Multi-Task Learning**: Concurrently learns pneumothorax lesion segmentation and auxiliary artifact detection to prevent false positives induced by surgical drains and skin folds.
2. **Hybrid Attention Architecture**: Incorporates spatial and channel attention mechanisms to capture fine edge boundaries of subtle apical pneumothoraces.
3. **Calibrated Operating Points**: Validation-tuned probability thresholds and minimum-component filtering to minimize noise and false positives in healthy scans.
