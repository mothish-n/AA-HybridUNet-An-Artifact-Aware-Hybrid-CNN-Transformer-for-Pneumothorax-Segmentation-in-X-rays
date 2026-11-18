# 04. Inference & Deployment

This folder contains the complete inference pipeline, benchmarking suite, and interactive Streamlit web application.

## Quick Launch (Streamlit Web Demo)
To start the research prototype web interface:
```bash
# Option 1: Double click run_app.bat on Windows
run_app.bat

# Option 2: Run via CLI
streamlit run app.py
```

## Contents
- **`app.py`**: Streamlit application allowing users to upload DICOM / PNG / JPEG chest X-rays, select models (`AA-HybridUNet` vs `Baseline U-Net`), adjust probability thresholds, toggle post-processing, and inspect overlays/probability maps.
- **`app_logic.py`**: Processing backend for image decoding, DICOM windowing/leveling, mask statistics, and overlay rendering.
- **`infer.py`**: Standalone inference module for loading deployment bundles, predicting images, and applying validation-tuned operating points.
- **`bench.py`**: Latency, throughput, and memory consumption benchmarking script across GPU and CPU.
- **`evalrun.py`**: Evaluation runner on test splits under clean and corrupted (severity 1-3) settings.
- **`metrics.py`**: Evaluation metrics implementation (Dice, IoU, Sensitivity, Specificity, Precision, Area Error).
- **`models/`**: Ready-to-deploy model weights and configs.
- **`requirements.txt`**: Python dependencies required for deployment.
