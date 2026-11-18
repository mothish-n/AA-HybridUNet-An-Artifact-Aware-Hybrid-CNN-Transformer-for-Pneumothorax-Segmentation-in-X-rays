# 02. Augmentation & Artifact Simulation

This folder contains the synthetic artifact simulation engine and data augmentation pipeline used to train and stress-test the Artifact-Aware Hybrid U-Net (AA-HybridUNet).

## Contents
- **`artifacts.py`**: Implementation of realistic medical imaging artifact injectors:
  - **Chest Tubes / Intercostal Drains**: Linear and curved radiopaque tubes with realistic attenuation.
  - **ECG Leads & Monitoring Wires**: Complex cable paths with high-contrast electrode attachments.
  - **Skin Folds**: Low-contrast linear pseudo-pneumothorax boundaries that commonly cause false positives.
  - **Surgical Hemostatic Clips**: Dense metallic artifacts with ray shadowing.
  - **Pacing Leads / Port-a-cath**: Intravascular catheters and generator artifacts.
  - **Noise & Blur**: High-frequency sensor noise and patient motion blur.
  - **Severity Levels**: Calibrated severity levels (Severity 1 to 3) and unseen test combinations.

## Visualizations
- `artifact_gallery.png`: Gallery of synthetic artifact types superimposed on representative radiographs.
- `artifact_head_output.png`: Intermediate feature maps and artifact-head probability predictions.
