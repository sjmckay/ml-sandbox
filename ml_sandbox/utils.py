import numpy as np
import matplotlib.pyplot as plt


def show_sample_images(img_dataset, num_images=5):
    """
    Show a grid of sample images from the given dataset.
    """
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


def show_images_with_predictions(images, true_labels, pred_labels, class_names=None):
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