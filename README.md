# Breast Cancer Histology Classification (BreaKHis): ResNet50 vs ResNet50+CBAM

This repo is the result of a few weeks of going back and forth on a binary
(benign vs malignant) histology image classifier trained on the
[BreaKHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis?select=BreaKHis_v1),
using a ResNet50 backbone fine-tuned in two variants a plain version, and
one with a CBAM (Convolutional Block Attention Module) block dropped in
after `layer4`. I'm writing this README the way the project actually went,
including the mistakes, because most of what I learned came from figuring
out *why* something didn't work rather than from the parts that worked on
the first try.

## The dataset problem I ran into (and it's the main thing worth reading)

BreaKHis has ~82 unique patients, each contributing multiple histology
images across 4 magnifications (40X, 100X, 200X, 400X). My first instinct
was to split at the image level, random 70/15/15 or similar. That's wrong
for this dataset: images from the same patient look very similar to each
other (same tissue, same staining batch), so if the same patient shows up
in both train and test, the model can partly "recognize" that patient
rather than genuinely generalize to new tissue. This is a well-known trap
with BreaKHis specifically, and a **lot of papers reporting 98-99% accuracy
on it are quietly doing image-level splits with this leakage baked in.**

So I switched to a **patient-grouped split** — every image belonging to a
given patient goes entirely into train, val, or test, never split across
them. I used `StratifiedGroupKFold` so the split still roughly preserves
the benign/malignant class ratio while respecting patient grouping.

Doing this the "correct" way exposed the real problem: **with only ~82
patients total, the resulting split is small and not very representative
of the full dataset.**

```
Train: 16 benign / 42 malignant patients  (58 patients)
Val:    3 benign /  5 malignant patients   (8 patients)
Test:   5 benign / 11 malignant patients  (16 patients)
```

8 patients in validation and 16 in test is not a lot to estimate
generalization from. On top of that, when I checked subtype breakdown
(BreaKHis has 4 benign and 4 malignant histological subtypes), some
subtypes were represented by literally 1-2 patients total, and
that was also a big issue cuz if you have 2 patients you cant put them in 3 categories
so one category ends up missing that subtype

None of this is a bug in the code, it's a genuine limitation of doing
patient-level grouping on a dataset this small. The honest takeaway is
that a single fixed split (like the one used here) should be read with
that problem in mind, and ideally validated further with patient-grouped
k-fold cross-validation rather than trusting one split's test numbers.
I didn't get to running full k-fold CV for this repo, but it's the
natural next step if I revisit this.

## Getting the training loop right

Along the way I hit a few of the classic fine-tuning mistakes, worth
listing since they cost real debugging time:

- **Optimizer scope bug**: when unfreezing the backbone for full
  fine-tuning after a frozen warmup phase, if the optimizer was created
  before unfreezing and was only given `model.fc.parameters()`, setting
  `requires_grad = True` on the backbone doesn't actually train it, the
  optimizer never sees those parameters. Fixed by building the optimizer
  over `model.parameters()` from the start.
- **Learning rate**: 1e-3 (fine for training a fresh head from scratch)
  is far too aggressive once you unfreeze a pretrained backbone, it
  wrecks the pretrained ImageNet features within a few epochs. Dropped to
  `3e-5` for the full-fine-tune phase.
- **Class imbalance**: BreaKHis skews malignant-heavy. Added
  class-weighted `CrossEntropyLoss` using `sklearn`'s
  `compute_class_weight("balanced", ...)`, computed from the train split
  only.
- **Scheduler**: went with `ReduceLROnPlateau` on validation F1 (`mode="max"`)
  rather than a fixed schedule like cosine annealing, since I wanted the
  LR to drop only when validation actually stopped improving.
- **Checkpointing on F1, not accuracy**  with class imbalance, accuracy is
  a misleading metric to select the best epoch on, so `best_model.pth` is
  saved on best validation F1 instead.

## Trying CBAM

After the baseline was training properly, I wanted to see if adding
attention would help, specifically CBAM, which does channel attention
(similar to Squeeze-and-Excitation block that i used in one of my other projects, but using both avg-pool and max-pool)
followed by spatial attention (which region of the feature map matters).
I inserted it right after `layer4`, before the average pooling.

I also tried only unfreezing `layer4` + the new head + CBAM during
fine-tuning, instead of unfreezing the entire backbone, since with
few patients I was worried about overfitting. That actually came out
*worse* than a full unfreeze (best val F1 ~0.937 vs ~0.958) apparently
layer1-3 do benefit from adapting to histology textures/staining rather
than staying frozen at ImageNet features, so I went back to unfreezing
everything after the warmup phase, same as the baseline.

## Results
Both models: ResNet50 (ImageNet-pretrained), warmup for 4 epochs (head +
CBAM, if present, only), then full fine-tuning for the rest,
Adam with weight decay `1e-4`, `lr=3e-5`, class-weighted loss,
`ReduceLROnPlateau` on val F1, checkpoint saved on best val F1, evaluated
once on the untouched test set at the end.

| Metric    | ResNet50 (baseline) | ResNet50 + CBAM |
|-----------|:--------------------:|:----------------:|
| Accuracy  | 0.8447               | **0.8464**       |
| Precision | **0.8261**           | 0.8188           |
| Recall    | 0.9360               | **0.9527**       |
| F1        | 0.8776               | **0.8807**       |
| ROC AUC   | 0.9315               | **0.9368**       |

CBAM edges out the baseline on most metrics, with a meaningfully higher
recall on the malignant class (0.9527 vs 0.9360) arguably the more
important direction to be strong in for a cancer screening task, since
missing a malignant case is worse than a false alarm on a benign one.
overall CBAM was the better model here, which is why
`evaluate.py` defaults to it.

Keep the dataset problem above in mind though, these are single-split
test numbers on a small, patient-grouped test set (16 patients, 1706
images), not a cross-validated estimate.

## Resnet50 + CBAM
- loss curve
- <img width="711" height="488" alt="image" src="https://github.com/user-attachments/assets/78f8a528-d9ec-4638-be14-48094b28d9dc" />
- Accuracy curve
- <img width="710" height="478" alt="image" src="https://github.com/user-attachments/assets/9d6703ee-10de-4186-a289-3538403b4267" />
- confusion matrix
- <img width="577" height="436" alt="image" src="https://github.com/user-attachments/assets/fe175a96-a51e-4f89-98b4-e1c516ca64fe" />


## Resnet50
- loss curve
- <img width="694" height="467" alt="image" src="https://github.com/user-attachments/assets/ea0cb12b-ab97-45b7-b04b-a3fa3eb6b379" />
- accuracy curve
- <img width="701" height="474" alt="image" src="https://github.com/user-attachments/assets/ce46d419-2286-4ebf-86fd-3decb0150201" />
- confusion matrix
- <img width="568" height="430" alt="image" src="https://github.com/user-attachments/assets/299a0fcd-fbed-44d0-8771-b4cb7064fb73" />


## Repo structure

```
.
├── dataset.py               # data loading, patient-grouped split, transforms, DataLoaders
├── model_resnet50.py        # baseline ResNet50 + custom head
├── model_resnet50_cbam.py   # ResNet50 + CBAM block after layer4 + custom head
├── train_only_resnet50.py    # training loop for simple resnet
├── train_resnet50_with_cbam.py # training loop for resnet +cbam
├── evaluate.py               # test-set evaluation, defaults to the CBAM model
├── resnet50_weights.pth      # trained baseline weights
├── resnet50_cbam_weights.pth # trained CBAM weights (best model)
└── requirements.txt
```

## Dataset setup

Download BreaKHis from Kaggle:
https://www.kaggle.com/datasets/ambarish/breakhis?select=BreaKHis_v1

Place it so the folder structure looks like:

```
dataset/BreaKHis_v1/histology_slides/breast/
    benign/SOB/...
    malignant/SOB/...
```

