import torch
from torch.utils.data import random_split
from .models import make_pca_rf_pipeline
from sklearn.model_selection import train_test_split

def train_eval_loop_skl(model, X, y, test_frac=0.2, random_state=42):
    """Train a Scikit-learn model on a given galaxy dataset.

    Args:
        model (PCARandomForestPipeline): The model to train.
        X (array-like): The input features.
        y (array-like): The target labels.
        test_frac (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        n_trees (int): Number of trees in the Random Forest.
        n_pca_comp (int): Number of components for PCA.
        **rf_kwargs: Additional keyword arguments for scikit-learn.ensemble.RandomForestClassifier.

    Returns:
        model (PCARandomForestPipeline): Trained model.
        test_accuracy (float): Accuracy on the test set.
        y_test (array-like): Test target labels.
        y_pred (array-like): Predicted labels for the test set.
    """

    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_state)   
    model.fit(X_train, y_train)
    # evaluate model on test set
    test_accuracy = model.score(X_test, y_test)
    return model, test_accuracy, y_test, model.predict(X_test)


def train_cnn(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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