import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Dataset: BreaKHis
# https://www.kaggle.com/datasets/ambarish/breakhis?select=BreaKHis_v1
DEFAULT_ROOT = "dataset/BreaKHis_v1/histology_slides/breast"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# extracts the data from dataset and stores it in dataframe with 
# columns: image_path,label,subtype of disease,patientid,magnification
def build_dataframe(root):
    data = []

    for label in ["benign", "malignant"]:

        label_path = os.path.join(root, label, "SOB")

        for subtype in os.listdir(label_path):

            subtype_path = os.path.join(label_path, subtype)

            if not os.path.isdir(subtype_path):
                continue

            for patient in os.listdir(subtype_path):

                patient_path = os.path.join(subtype_path, patient)

                if not os.path.isdir(patient_path):
                    continue

                for magnification in os.listdir(patient_path):

                    mag_path = os.path.join(patient_path, magnification)

                    if not os.path.isdir(mag_path):
                        continue

                    for image in os.listdir(mag_path):

                        if image.endswith(".png"):
                            data.append({
                                "image_path": os.path.join(mag_path, image),
                                "label": label,
                                "subtype": subtype,
                                "patient_id": patient,
                                "magnification": magnification
                            })

    df = pd.DataFrame(data)

    df["label"] = df["label"].map({
        "benign": 0,
        "malignant": 1
    })

    return df


def split_dataset(df):
    # Patient-grouped, stratified(using labels) split. Splitting is done at the patient
    # level (not image level) so that no patient's images (actually each patient has multiple images of diff magnifications) 
    # end up in more than one of train / val / test otherwise the model can leak
    # patient-specific staining/texture info across the split.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # take one fold as train+val vs test (80/20-ish)
    train_val_idx, test_idx = next(
        sgkf.split(df, y=df["label"], groups=df["patient_id"])
    )

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # split train_val into train/val the same way
    sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)  # ~12.5% val
    train_idx, val_idx = next(
        sgkf2.split(train_val_df, y=train_val_df["label"], groups=train_val_df["patient_id"])
    )

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class BreastCancerDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(root=DEFAULT_ROOT, batch_size=32, num_workers=None):
    set_seed(42)

    df = build_dataframe(root)
    train_df, val_df, test_df = split_dataset(df)

    train_dataset = BreastCancerDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = BreastCancerDataset(
        val_df,
        transform=test_transform
    )

    test_dataset = BreastCancerDataset(
        test_df,
        transform=test_transform
    )

    if num_workers is None:
        num_workers = min(os.cpu_count(), 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
