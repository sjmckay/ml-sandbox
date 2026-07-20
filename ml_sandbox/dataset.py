import os
import numpy as np
import matplotlib.pyplot as plt
from galaxy_datasets import gz_candels, gz2
from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.model_selection import train_test_split
import multiprocessing

def load_catalog(dir='./.tmp', dataset_name : str = 'candels'):   
    """Load the catalog, downloading it if necessary."""
    if dataset_name not in ['candels', 'gz2']: 
        raise ValueError(f"Invalid dataset_name '{dataset_name}'. Must be one of ['candels', 'gz2'].")
    if dataset_name == 'candels': dataset = gz_candels
    elif dataset_name == 'gz2': dataset = gz2
    try:
        catalog, label_cols = dataset(
            root=dir,
            download=False,
        )
    except:
        catalog, label_cols = dataset(
            root=dir,
            download=True,
        )
    return catalog, label_cols


from PIL import Image
def load_image(filename, dir='./.tmp'):
    img_path = os.path.join(dir, 'images', filename)
    image = Image.open(img_path)
    return image


def d8_random(img):
    """Apply a random transformation from the D8 group to the input image (tensor), for improved training.
    Modified version of function by John Wu, NASA STIG AI lectures"""
    choice = np.random.randint(0, 7)
    if choice == 0:
        return img                              # identity
    elif choice == 1:
        return transforms.functional.rotate(img, 90, expand=True)
    elif choice == 2:
        return transforms.functional.rotate(img, 180, expand=True)
    elif choice == 3:
        return transforms.functional.rotate(img, 270, expand=True)
    elif choice == 4:
        return transforms.functional.hflip(img) # horizontal reflection
    elif choice == 5:
        return transforms.functional.vflip(img) # vertical reflection
    elif choice == 6:
        return torch.transpose(img, -2, -1)   # main diagonal
    elif choice == 7:
        return torch.transpose(img, -2, -1)[::-1]   # anti-diagonal

class RandomD8(transforms.RandomApply):
    def __init__(self, p=1.0):
        super().__init__([transforms.Lambda(d8_random)], p=p)


def assign_label_by_max_vote(row):
    """Given a row of the catalog, return the class label assigned the most votes."""
    return row.argmax()


def get_dataset(catalog, label_keys: str, size=None, train=True, resize = 32, dir='./.tmp'):
    """Create a PyTorch Dataset from the given catalog, applying necessary transforms and augmentations.
    
    Args:
        catalog (pd.DataFrame): The input catalog containing image filenames and labels.
        label_key (str): The prefix of the label columns to use for classification.
        size (int, optional): If specified, limits the dataset to this many samples. Defaults to None (use all).
        train (bool, optional): Whether this dataset is for training (applies augmentations) or testing. Defaults to True.
        dir (str, optional): Directory where images are stored. Defaults to './.tmp'.
    """
    subset = size//10 if size is not None and size > 10 else catalog.shape[0] // 10

    if train: transforms_to_use = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((resize, resize)),
                    RandomD8(),
                    ])
    else: transforms_to_use = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((resize, resize)),
                    ])
    #pick out label columns that start with the specified key (e.g. 'merging-candels') to use as targets
    dataset = GalaxyDataset(
        catalog=catalog.sample(size) if size is not None else catalog,
        label_cols=label_keys,
        transform=transforms_to_use,
        target_transform=transforms.Lambda(assign_label_by_max_vote),
    )
    return dataset


def get_dataloader(dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=True):
    """
    Create a PyTorch DataLoader from the given dataset, with specified batch size and number of workers.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _collate_fn(batch):
    imgs = [b[0].cpu().numpy().ravel() for b in batch]
    labels = [int(b[1].item()) if hasattr(b[1], "item") else int(b[1]) for b in batch]
    return np.stack(imgs), np.array(labels, dtype=int)


def convert_dataset_sklearn(dataset, batch_size=32, num_workers=None):
    """
    Convert a torch-style dataset to (X, y) using parallel workers.

    Uses a PyTorch `DataLoader` with `num_workers` to parallelize item loading
    and transforms. This avoids pickling the whole dataset object for
    multiprocessing and leverages DataLoader's worker pool.
    """

    if num_workers is None:
        num_workers = min(4, multiprocessing.cpu_count())

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_fn)

    X_parts = []
    y_parts = []
    for Xb, yb in loader:
        X_parts.append(Xb)
        y_parts.append(yb)

    if len(X_parts) == 0:
        return np.empty((0,)), np.empty((0,))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y

