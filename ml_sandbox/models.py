import torch.nn as nn
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier


class SimpleCNNClassifier(nn.Module):
    """A simple Convolutional Neural Network (CNN) classifier for image data.
    """
    def __init__(self, input_channels=1, num_classes=2, side=64):
        super(SimpleCNNClassifier, self).__init__()
        self.side = side
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (side // 4) * (side // 4), 128)  # Assuming input image size is side x side
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * (self.side // 4) * (self.side // 4))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    


class PCARandomForestPipeline(RandomForestClassifier):
    """A pipeline that first applies PCA for dimensionality reduction
    and then fits a Random Forest classifier.
    """
    def __init__(self, n_components=150, **rf_kwargs):
        super(PCARandomForestPipeline, self).__init__(**rf_kwargs)
        self.pca = RandomizedPCA(n_components=n_components)

    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        return super().fit(X_reduced, y)

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        return super().predict(X_reduced)
    
    def predict_proba(self, X):
        X_reduced = self.pca.transform(X)
        return super().predict_proba(X_reduced)
