import torch
from torch.utils.data import random_split
from .models import make_pca_rf_pipeline
from sklearn.model_selection import train_test_split
import numpy as np

def train_cnn(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device,non_blocking=True)
        y = y.to(device,non_blocking=True)

        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_i, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_i:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= len(dataloader)
    correct /= size
    return train_loss, correct # return loss and accuracy for the epoch


def test_cnn(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device,non_blocking=True)
            y = y.to(device,non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct # return loss and accuracy for the epoch


def plot_training_history(history):
    """Utility function to plot training and testing loss and accuracy over epochs."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_images_with_predictions(images, true_labels, pred_labels, class_names=None):
    """Utility function to plot images with their true and predicted labels."""
    import matplotlib.pyplot as plt
    num_images = len(images)
    grid_x = int(np.ceil(np.sqrt(num_images)))
    grid_y = int(np.ceil(num_images / grid_x))
    
    plt.figure(figsize=(grid_x*3, grid_y*3))
    for i in range(num_images):
        plt.subplot(grid_y, grid_x, i+1)
        plt.imshow(images[i].permute(1,2,0))  # Assuming images are in (C,H,W) format
        true_label = class_names[true_labels[i]] if class_names else true_labels[i]
        pred_label = class_names[pred_labels[i]] if class_names else pred_labels[i]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
