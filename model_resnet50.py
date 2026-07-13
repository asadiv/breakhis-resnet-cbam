from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model():
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)
     # replace the fc layer fo resnet50, as it was trained for 1000 classes ad gives 1000 outputs.
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)   #we only have two classes (benign, malignant)
    )

    return model
