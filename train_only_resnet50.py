import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from tqdm import tqdm
from dataset import DEFAULT_ROOT, get_dataloaders
from model_resnet50 import build_model as build_without_cbam
from model_resnet50_cbam import build_model as build_cbam


def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in tqdm(dataloader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_without_cbam()
    model = model.to(device)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        DEFAULT_ROOT, batch_size=32
    )

    # the dataset is imbalanced so we are gonna punish the minority class more
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # because we're monitoring F1
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    NUM_EPOCHS = 30

    best_f1 = 0

    # warmup phase: only train the newly added head
    # while keeping the pretrained backbone frozen
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    for epoch in range(NUM_EPOCHS):
        if epoch == 4:
            # unfreeze the whole backbone for full fine-tuning
            for param in model.parameters():
                param.requires_grad = True

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Train Acc: {train_acc:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Val Acc: {val_acc:.4f}"
            f" | Val F1: {val_f1:.4f}"
        )


if __name__ == "__main__":
    main()
