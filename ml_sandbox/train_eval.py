import torch
from torch.utils.data import random_split
from .models import make_pca_rf_pipeline
from sklearn.model_selection import train_test_split

def train_skl(X_train, y_train, model, random_state=42):
    """Train a Scikit-learn model on a given galaxy dataset.

    Args:
        model (PCARandomForestPipeline): The model to train.
        X_train (array-like): Training features.
        y_train (array-like): Training target labels.
        random_state (int): Random seed for reproducibility.

    """



def train_cnn(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_cnn(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")