import torch.nn as nn
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline


class SimpleCNNClassifier(nn.Module):
    """A simple Convolutional Neural Network (CNN) classifier for image data with one or more channels and variable size.
    """
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)
    

class TransformerClassifier(nn.Module):
    """A Transformer-based classifier for image data with one or more channels and variable size.
    """
    def __init__(self, input_channels=1, num_classes=2, num_heads=4, num_layers=2, dim_feedforward=128):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.project = nn.Linear(32 * 28 * 28, 256)  # Project to a fixed size for the transformer

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.project(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)  # Add sequence dimension
        return self.classifier(x)
    

def make_pca_rf_pipeline(n_pca_comp=150, n_trees=100, random_state=42, **rf_kwargs):
    pca = PCA(n_components=n_pca_comp, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, **rf_kwargs)
    return make_pipeline(pca, rf)