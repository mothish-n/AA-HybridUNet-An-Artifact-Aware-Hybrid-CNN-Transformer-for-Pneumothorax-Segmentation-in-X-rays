# 03. Model Architecture & Training

This folder contains the network architectures, loss formulations, training engines, and experimental logs for both the proposed AA-HybridUNet and the baseline model.

## Contents
- **`models/`**:
  - `aa_hybrid_unet.py`: The proposed Artifact-Aware Hybrid U-Net featuring dual-head architecture (pneumothorax segmentation + auxiliary artifact classification/localization) and attention-guided skip connections.
  - `baseline_unet.py`: Standard EfficientNet-B4 U-Net baseline for benchmark comparison.
- **`losses.py`**: Combined loss formulations:
  - BCE + Soft Dice Loss for positive/negative mask balance.
  - Auxiliary Artifact Loss (multi-task objective enabling robust representation learning).
- **`engine.py`**: Training loop with mixed-precision (FP16), AdamW optimizer, cosine annealing learning rate scheduler, checkpointing, and validation hooks.
- **`checkpoints/`**: Best saved weights (`weights.pt`), model configs (`model_config.json`), and operating points (`operating_point.json`).
- **`runs/`**: Full epoch-by-epoch training logs (`history.csv`), run configs, and efficiency benchmarks.

## Visualizations
- `training_curves.png`: Loss progression, validation Dice, sensitivity, and precision over training epochs.
- `training_summary.csv`: Summary table of convergence epochs, runtime per epoch, and best checkpoints.
