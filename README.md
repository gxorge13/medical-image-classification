# Image Classification with Transfer Learning

An **APS360 team project at the University of Toronto** exploring three-class image classification with PyTorch. The pipeline uses VGG16 feature extraction, a fully connected classifier, and an SVM baseline, with scripts for training and evaluation.

**Stack:** Python · PyTorch · torchvision · NumPy · scikit-learn · Matplotlib

## Project overview

The data loader selects three dataset labels: `ProcessedCancer`, `Normal`, and `COVID`. It balances the selected classes, resizes images to 299 × 299, and creates approximately 70% / 15% / 15% train, validation, and test subsets.

The transfer-learning path extracts **VGG16 convolutional features** and feeds them into a classifier with batch normalization and dropout. Separate scripts explore an SVM baseline and report classification metrics and confusion matrices.

This is an educational image-classification experiment, not a clinically validated diagnostic tool.

## Explore the implementation

| File | Purpose |
| --- | --- |
| [src/helper.py](src/helper.py) | Data preparation, class balancing, dataset splits, training helpers, and evaluation |
| [src/transfer_learning.py](src/transfer_learning.py) | VGG16 feature extraction and feature storage |
| [src/classifier.py](src/classifier.py) | Fully connected classification network |
| [src/svm_baseline.py](src/svm_baseline.py) | SVM baseline experiment |
| [src/confusion.py](src/confusion.py) | Confusion matrices, precision, recall, F1, and accuracy |
| [src/classify.py](src/classify.py) and [src/test.py](src/test.py) | Inference and evaluation using saved weights |

## Setup

The repository includes a large `Data/` directory. To inspect the code without downloading the image collection, browse `src/` on GitHub first.

For a local checkout:

```bash
git clone https://github.com/gxorge13/medical-image-classification.git
cd medical-image-classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scikit-learn
```

On Windows, use `.venv\Scripts\activate` for activation. The extra scikit-learn dependency is needed by the baseline and confusion-matrix scripts but is not listed in the original requirements file. Dependency versions are unpinned.

Run scripts from the **repository root** so the relative `Data/` and `models/` paths resolve correctly. VGG16 initialization may download pretrained weights.

## Reproducing the experiments

The repository preserves development scripts rather than a single turnkey training command:

1. Check that `Data/` contains the expected class folders and that the intended dataset split is appropriate for the experiment.
2. In `src/transfer_learning.py`, the active `splits` mapping currently extracts **validation and test features only**. Include the training loader there when generating features for a fresh classifier-training run, then use `python src/main.py` to extract features.
3. `src/train.py` currently plots an existing training curve; its training call is commented out. Configure that call and the output paths before training a fresh classifier.
4. Update the saved-model paths in `src/test.py`, `src/classify.py`, or `src/confusion.py` to match the weights from the intended run before evaluation.

Feature tensors and trained classifier weights under `models/` are not included in the root checkout. The full training/evaluation workflow has not been reproduced during this documentation cleanup.

## Results and limitations

The source includes evaluation routines and a historical metric comment, but there is no reproduced benchmark report accompanying this README. Results should be reported with the exact data split, preprocessing, checkpoint, and evaluation procedure. The current split operates on images; it does not establish patient-level separation.

## Team and credit

Repository contributors include [George Gerges](https://github.com/gxorge13), [SelimAbdelwahab](https://github.com/SelimAbdelwahab), [youssefhsaad](https://github.com/youssefhsaad), and [mazismail](https://github.com/mazismail).

The features described here belong to the combined team project. Individual component ownership is not inferred from commit counts.
