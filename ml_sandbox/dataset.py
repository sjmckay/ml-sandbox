import numpy as np


def load_data_set(target='galaxy_mnist', dir='./.tmp', framework='torch'):
    if target == 'galaxy_mnist':
        from galaxy_mnist import GalaxyMNIST
        catalog, labels = GalaxyMNIST(
            root=dir,
            download=True,
            train=True  # by default, or set False for test set
        )
    elif target == 'gz_candels':
        from galaxy_datasets import gz_candels
        catalog, labels = gz_candels(
            root=dir,
            download=True,
            train=True  # by default, or set False for test set
        ) 
    else: raise ValueError(f"Unknown dataset target: {target}")

    if framework == 'torch':
        from galaxy_datasets.pytorch.galaxy_dataset import CatalogDataset
        dataset = CatalogDataset(
            catalog=catalog,
            label_cols=labels.columns.tolist(),
        )
    else: raise ValueError(f"Unknown framework: {framework}")

    return dataset



def show_sample_images(dataset, num_images=5):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if image.shape[0] == 1:  # grayscale
            image = image.squeeze(0)
        elif image.shape[0] == 3:  # RGB
            image = np.transpose(image, (1, 2, 0))
        axes[i].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

