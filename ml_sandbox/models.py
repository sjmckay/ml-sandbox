import torch.nn as nn
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline


class SimpleCNNClassifier(nn.Module):
    """A simple Convolutional Neural Network (CNN) classifier for image data.
    """
    def __init__(self, input_channels=1, num_classes=1, side=64):
        super(SimpleCNNClassifier, self).__init__()
        self.side = side
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Compute the flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, side, side) # run dummy image through conv to determine flattened size for fully connected layer
            dummy = self.pool(self.relu(self.conv1(dummy)))
            dummy = self.pool(self.relu(self.conv2(dummy)))
            self.flat_features = dummy.numel()
        
        self.fc1 = nn.Linear(self.flat_features, 128)  # Assuming input image size is side x side
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flat_features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

def make_pca_rf_pipeline(n_pca_comp=150, n_trees=100, random_state=42, **rf_kwargs):
    pca = PCA(n_components=n_pca_comp, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, **rf_kwargs)
    return make_pipeline(pca, rf)