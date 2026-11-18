# 05. Visualization & Reporting

This folder contains the complete suite of qualitative and quantitative evaluation results, publication-ready figures, and LaTeX/Markdown/CSV result tables.

## Contents
- **`figures/`**:
  - `qualitative_clean.png`: Visual comparisons of predicted masks vs ground truth on clean test radiographs.
  - `qualitative_sev2.png`: Visual comparisons under synthetic artifact contamination (Severity 2).
  - `robustness.png`: Robustness degradation curves comparing AA-HybridUNet vs Baseline under escalating artifact levels.
  - `roc_pr_curves.png`: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
  - `val_threshold_sweep.png`: Threshold sweep curves for post-processing optimization.
- **`tables/`**:
  - `table1_main_clean.*`: Main benchmark results on clean test split (CSV, MD, LaTeX).
  - `table2_robustness_dice_pos.*`: Robustness benchmark on lesion-positive cases.
  - `table2b_robustness_dice_all.*`: Robustness benchmark across all test cases.
- **`test_results/` & `val_results/`**: Per-image predictions, Dice scores, and summary JSONs across all test conditions (clean, sev1, sev2, sev3, unseen).
- **`report.py`**: Automated script to compile summaries and format LaTeX/Markdown tables.
- **`measured_speed.csv`**: Inference latency (ms/image) and throughput across hardware targets.
