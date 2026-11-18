# 01. Data Pipeline

This folder manages the data ingestion, dataset splitting, pre-processing, and caching pipeline for chest X-ray pneumothorax segmentation.

## Contents
- **`dataset.py`**: PyTorch Dataset definitions, image loaders (supporting 16-bit DICOM, PNG, JPEG), CLAHE contrast adjustment, resizing, and ground-truth RLE/binary mask extraction.
- **`cache.py`**: High-performance disk/RAM caching utilities to accelerate multi-epoch training.
- **`split_manifest.csv`**: Full metadata manifest recording patient IDs, image paths, ground truth presence, and split assignments (`train`, `val`, `test`).
- **`split_info.json`**: Exact summary of train/val/test patient counts and positive/negative balance.
- **`dataset_audit.csv`**: Comprehensive audit of original image dimensions, aspect ratios, dynamic range, and label distributions.

## Visualizations
- `dataset_distribution.png`: Class balance and patient distributions across splits.
- `dataset_samples.png`: Representative frontal chest X-ray samples with annotated pneumothorax regions.
- `lesion_size.png`: Size distribution analysis of pneumothorax lesions across the dataset.
