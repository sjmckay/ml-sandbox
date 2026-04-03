import numpy as np
import matplotlib.pyplot as plt
import torch

from galaxy_datasets import gz_candels 
from ml_sandbox.dataset import get_dataset
from ml_sandbox.models import SimpleCNNClassifier
from ml_sandbox.train_eval import train_cnn, test_cnn
from ml_sandbox.dataset import get_dataloader
from sklearn.model_selection import train_test_split


### INPUTS
num_epochs = 5
label_key = 'merging-candels'
###

if __name__ == '__main__':
    catalog, label_cols = gz_candels(
            root='.tmp/',
            train=True,
            download=True,
    )



    model = SimpleCNNClassifier(num_classes=np.sum(catalog.columns.str.startswith(label_key)),
                                input_channels=3)

    train_cat, test_cat = train_test_split(catalog, test_size=0.2, random_state=42)

    train_dataset = get_dataset(train_cat, label_key=label_key, 
                                size=train_cat.shape[0]//5, 
                                train=True, dir='.tmp/')
    test_dataset = get_dataset(test_cat, label_key=label_key, 
                               size=test_cat.shape[0]//5, 
                               train=False, dir='.tmp/')

    train_loader = get_dataloader(train_dataset, batch_size=8)
    test_loader = get_dataloader(test_dataset, batch_size=8)

    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()

    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    hist = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    do_cnn_training = True
    if do_cnn_training:
        for i in range(num_epochs):
            train_loss, train_acc = train_cnn(train_loader, model, criterion, optimizer, device=device) 
            test_loss, test_acc = test_cnn(test_loader, model, criterion, device=device)
            hist['train_loss'].append(train_loss)
            hist['train_acc'].append(train_acc)
            hist['test_loss'].append(test_loss)
            hist['test_acc'].append(test_acc)
            print(f"Epoch: {i+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        from ml_sandbox.train_eval import plot_training_history
        plot_training_history(hist)