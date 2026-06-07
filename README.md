# Distorted Visual Sequence Pattern Recognition using Deep Learning

## Problem Statement

Many modern digital systems need to read text from images that are unclear, noisy, or intentionally
distorted. In real-world situations, characters may overlap with each other, contain random noise, may be
partially hidden, or appear in unusual shapes and positions. These challenges make normal text
recognition systems unreliable.

In this challenge, the goal is to build a deep learning model that can correctly recognize and
reconstruct text sequences from visually distorted grayscale images.

## Objective
Given a distorted  image containing a hidden text sequence, the model must predict the
correct sequence of characters present in the image.

## Dataset Structure
- **Training Set** — 20,000 images with corresponding text labels
- **Test Set** — 5,000 images with no labels (predictions must be generated)

## Evaluation Metric
Submissions are evaluated using **Character Error Rate (CER)** based on Levenshtein Distance —
the minimum number of insertions, deletions, and substitutions required to transform the predicted
text into the correct text. Lower CER = better performance.

## Results
| Metric | Value |
|--------|-------|
| Validation CER | 0.0001 |
| Exact Match Accuracy | 99.95% |
| Correct Predictions | 1999 / 2000 |

## Model Architecture — CRNN
```
Input (64 x 200 x 1)
      CNN (5 blocks — Conv-BN-ReLU)
Feature Map (1 x 50 x 256)
      Reshape
Sequence (50 time-steps x 256 features)
      BiLSTM x 2 (128 units each direction)
      Dense + Softmax
Output (50 x 32)  →  CTC Loss
```

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib (plots)
- Pillow (image loading)
- tf.data pipeline with AUTOTUNE prefetching
- Mixed Precision (float16)
- editdistance (CER metric)
- scikit-learn (train/val split)

## Project Structure
```
Text_detection/
├── README.md
├── distortedpattern_Recognition.ipynb
└── data/cig_ps/
    ├── train_images/
    ├── test_images/
    ├── train-labels.csv
    └── outputs/
        ├── best_crnn_v3.keras
        ├── submission_Mona_24113087.csv
        ├── training_curves.png
        ├── val_predictions.png
        └── training_log.csv
```

## Dataset

The dataset is not included in this repository due to size constraints.
Download it from Google Drive and place it in `data/cig_ps/`:

https://drive.google.com/file/d/1e0Bpbmjp-Pc1Oz7luC1g8JhfzKpGZRHu/view?usp=drive_link

The dataset contains:
- `train_images/` — 20,000 distorted images
- `test_images/`  — 5,000 distorted images
- `train-labels.csv` — image filenames and corresponding text labels