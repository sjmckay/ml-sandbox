import numpy as np


def load_data_set(target='gz_candels', dir='./.tmp'):
    if target == 'gz_candels':
        from galaxy_datasets.pytorch import GZCandels
        target_class = GZCandels
    else: raise ValueError(f"Unknown dataset target: {target}")
   
    return target_class(
        root=dir,
        download=True,
        train=True  # by default, or set False for test set
    ) 



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

