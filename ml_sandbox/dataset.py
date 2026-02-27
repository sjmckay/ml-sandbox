import os
import numpy as np
import matplotlib.pyplot as plt
from galaxy_datasets import gz_candels
from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.model_selection import train_test_split


def load_catalog(dir='./.tmp'):   
    try:
        catalog, label_cols = gz_candels(
            root=dir,
            download=False,
        )
    except:
        catalog, label_cols = gz_candels(
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

def get_dataset(catalog, label_cols, size=1000, train=True, dir='./.tmp'):
    subset = size//10 if size > 10 else size
    ims = torch.stack([ToTensor()(load_image(catalog['filename'][i], dir=dir)) for i in range(subset)])
    img_mean = ims.mean(dim=(0,2,3))
    img_std = ims.std(dim=(0,2,3))
    if train: transforms_to_use = transforms.Compose([
                    transforms.ToTensor(),
                    RandomD8(),
                    transforms.Normalize(mean=img_mean, std=img_std),])
    else: transforms_to_use = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=img_mean, std=img_std)])
    dataset = GalaxyDataset(
        catalog=catalog.sample(size) if size is not None else catalog,
        label_cols=label_cols,
        transform=transforms_to_use,
        target_transform=lambda x: torch.tensor(x, dtype=torch.float32),
    )
    return dataset


def get_dataloader(dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def show_sample_images(img_dataset, num_images=5):
    grid_x = int(np.ceil(np.sqrt(num_images)))
    grid_y = int(np.ceil(num_images / grid_x))
    fig, axes = plt.subplots(grid_y, grid_x, figsize=(grid_x*3, grid_y*3))
    axes = axes.flatten() if num_images > 1 else [axes]
    for i in range(num_images):
        index = np.random.randint(len(img_dataset))
        image, _ = img_dataset[index]
        try: axes[i].set_title(f"{img_dataset.catalog.loc[index,'filename'].split('.jpg')[0]}")
        except: axes[i].set_title(f"Image {index}")
        try:
            axes[i].imshow(image)
        except TypeError:
            from torchvision.transforms import ToPILImage
            axes[i].imshow(ToPILImage()(image))
    for ax in axes:
        ax.axis('off')
    plt.show()


def convert_dataset_sklearn(dataset):
    X = []
    y = []
    from multiprocessing import Pool, cpu_count
    def process_item(i):
        img, label = dataset[i]
        return img.numpy().flatten(), label.item()
    with Pool(cpu_count()) as p:
        results = p.map(process_item, range(len(dataset)))
    X, y = zip(*results)
    return np.array(X), np.array(y)
