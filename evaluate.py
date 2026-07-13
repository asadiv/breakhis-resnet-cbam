import argparse
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from dataset import DEFAULT_ROOT, get_dataloaders
from model_resnet50 import build_model as build_without_cbam    
from model_resnet50_cbam import build_model as build_cbam


def evaluate(model, dataloader, device):

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(all_labels, all_preds))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }


def main():
    # or you could use the resnet50 without cbam aswell, then build the respective model for it
    weights = "resnet50_cbam_weights.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _, _, _ = get_dataloaders(DEFAULT_ROOT)

    # you can build the simple resnet50 without cbam aswell, then use the respective weights
    model = build_cbam()
    model = model.to(device)
    model.load_state_dict(torch.load(weights, map_location=device))

    metrics = evaluate(model, test_loader, device)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=["Benign", "Malignant"]
    )

    disp.plot(cmap="Blues")
    plt.show()


if __name__ == "__main__":
    main()
