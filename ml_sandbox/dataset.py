import os
import numpy as np
import matplotlib.pyplot as plt
from galaxy_datasets import gz_candels
from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset
from torch.utils.data import DataLoader


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
    image = Image(img_path)
    return image

def get_sklearn_dataset(catalog, label_cols):
    X = [load_image(fname) for fname in catalog['filename']]
    y = catalog[label_cols]
    return X, y


def load_dataset(dir='./.tmp', size=1000, labels=None):
    catalog, _ = load_catalog(dir=dir)

    dataset = GalaxyDataset(
        catalog=catalog.sample(size) if size is not None else catalog,
        label_cols=labels if labels is not None else catalog.columns.tolist()[:-4],
        transform=None,
    )

    return dataset

def get_dataloader(dataset, batch_size=32, num_workers=os.cpu_count()):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return dataloader


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
        axes[i].imshow(image)
    for ax in axes:
        ax.axis('off')
    plt.show()

