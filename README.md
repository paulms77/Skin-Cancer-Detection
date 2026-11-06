# Skin Cancer Detection with 3D-TBP

Multimodal classifier for the **ISIC 2024 Challenge** — detecting histologically
confirmed skin cancer in single-lesion crops taken from 3D total-body photography
(3D-TBP), combining a CNN over the lesion image with a neural network over
clinical metadata.

| | |
| --- | --- |
| **Competition** | [ISIC 2024 — Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge) |
| **Task** | Binary classification — malignant vs. benign |
| **Metric** | pAUC — partial AUC above 80% TPR |
| **Paper** | [Skin Cancer Classification using Deep Learning](https://drive.google.com/file/d/1hsKeB9cscI6wswlgyTnQX5Il1fZC6LpP/view) |

---

## The data

Severely imbalanced, which defines the whole problem:

| | Count |
| --- | --- |
| Benign images | 400,666 |
| **Malignant images** | **393** |
| Metadata records | 501,059 rows × 55 fields |

Roughly **1 in 1,000**. A model that predicts "benign" every time is 99.9%
accurate and clinically worthless.

Data is large, so the practical route is a Kaggle API key, downloading the
`.zip` and unpacking it on Drive rather than pulling files individually.

<img width="400" alt="Class distribution" src="https://github.com/user-attachments/assets/7bf0e0db-e80b-42d3-b862-b05e236525ff">

## Evaluation — why pAUC

<img width="300" alt="pAUC" src="https://github.com/user-attachments/assets/7765dd15-1024-4e03-b5d8-190cee71a1a4">

Scoring uses the **partial** area under the ROC curve, restricted to the region
above 80% true-positive rate.

This is the right metric for the clinical question. A screening tool that misses
one malignant lesion in five is unusable regardless of how cleanly it sorts the
easy benign cases, so the metric simply ignores the low-sensitivity part of the
curve. Optimising full AUC here would reward the wrong behaviour.

## Pipeline

### Image preprocessing

Three artefact classes are removed before anything else, because each gives the
network a shortcut that does not generalise:

- **Hair removal** — dark linear structures that occlude the lesion
- **Background removal** — skin surrounding the lesion
- **Marker / ink removal** — clinician pen marks

The ink matters most. Clinicians mark lesions they are *already suspicious of*,
so ink correlates with malignancy in the training set. A model left to find that
correlation learns to detect pen, not cancer, and collapses on clean images.

| Before | After |
| --- | --- |
| <img width="420" src="https://github.com/user-attachments/assets/a606d5aa-b9a4-44c7-8ae2-76ad42a0ec46"> | <img width="420" src="https://github.com/user-attachments/assets/e931e572-e5e2-4d56-99f7-b175b772671b"> |

### Metadata processing

- **Feature engineering** — 44 additional derived indicators
- Missing-value handling, categorical encoding, numeric scaling
- **Adversarial validation** — train a classifier to distinguish train rows from
  test rows. If it succeeds, the two distributions differ and local validation
  will not predict the leaderboard. Features that leak that distinction get
  dropped.

### Augmentation

Applied to counter the imbalance and the narrow imaging conditions:

`Transpose` · `RandomBrightnessContrast` · `CoarseDropout` ·
`MotionBlur` / `MedianBlur` / `GaussianBlur` ·
`OpticalDistortion` / `GridDistortion` / `ElasticTransform` ·
`ShiftScaleRotate` · `CLAHE` · `HueSaturationValue` · `Microscope` ·
`Normalize`

<img width="420" alt="Augmentation result" src="https://github.com/user-attachments/assets/36f67863-107a-41b1-b736-331debb74a2b">

## Model

A two-branch network: **EfficientNet-B3** over the lesion crop, a dense network
over the engineered metadata, with the two representations concatenated before
the classification head.

<img width="420" alt="Architecture" src="https://github.com/user-attachments/assets/f44dc39d-8ef7-4fe3-9bf6-a5041d5acd20">

Both branches carry real signal. The image shows what the lesion looks like; the
metadata carries patient age, anatomical site, and — importantly — how this
lesion compares to *the same patient's other lesions*, which is the "ugly
duckling" heuristic dermatologists use directly.

## Setup

```bash
git clone https://github.com/paulms77/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection

pip install polars pandas numpy timm torch scikit-learn albumentations \
            h5py seaborn matplotlib
```

Fetch the data with the Kaggle API:

```bash
pip install kaggle
kaggle competitions download -c isic-2024-challenge
unzip isic-2024-challenge.zip -d data/
```

## Usage

```bash
jupyter notebook ISIC2024.ipynb
```

Runs end to end: metadata loading with Polars, image artefact removal, feature
engineering, adversarial validation, augmented training of the multimodal
network, and submission assembly. A GPU is effectively required for the image
branch.

## Results

**pAUC ≈ 0.167** on the `[0.0, 0.2]` scale after a single epoch.

That is one epoch, reported as such. The full-length runs and their analysis are
in the [paper](https://drive.google.com/file/d/1hsKeB9cscI6wswlgyTnQX5Il1fZC6LpP/view).

## Dependencies

[Polars](https://pola.rs/) ·
[Pandas](https://pandas.pydata.org/) ·
[NumPy](https://numpy.org/) ·
[timm](https://timm.fast.ai/) ·
[PyTorch](https://pytorch.org/) ·
[scikit-learn](https://scikit-learn.org/stable/) ·
[Albumentations](https://albumentations.ai/docs/) ·
[h5py](https://docs.h5py.org/en/stable/) ·
[Seaborn](https://seaborn.pydata.org/) ·
[Matplotlib](https://matplotlib.org/)

## License

MIT — see [LICENSE](LICENSE).
