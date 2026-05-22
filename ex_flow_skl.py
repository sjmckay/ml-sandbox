from ml_sandbox.dataset import convert_dataset_sklearn
from ml_sandbox.models import make_pca_rf_pipeline
import numpy as np
import matplotlib.pyplot as plt

from galaxy_datasets import gz_candels 
from ml_sandbox.dataset import get_dataset
from ml_sandbox.models import SimpleCNNClassifier
from ml_sandbox.dataset import get_dataloader
from sklearn.model_selection import train_test_split
from ml_sandbox.utils import show_images_with_predictions


### INPUTS
label_key = 'spiral-arm-count-candels'
###

if __name__ == '__main__':
    print('Loading catalog...')
    catalog, label_cols = gz_candels(
            root='.tmp/',
            train=True,
            download=True,
    )
    print(f'Catalog loaded with {len(catalog)} entries and label columns: {label_cols}')

    # split into training and testing/validation
    print('Splitting catalog into training and testing sets...')
    train_cat, test_cat = train_test_split(catalog, test_size=0.2, random_state=42)
    train_dataset = get_dataset(train_cat, label_key=label_key, 
                                size=train_cat.shape[0]//5, 
                                train=True, dir='.tmp/')
    test_dataset = get_dataset(test_cat, label_key=label_key, 
                               size=test_cat.shape[0]//5, 
                               train=False, dir='.tmp/')
    print(f'Training set: {len(train_dataset)} samples, Testing set: {len(test_dataset)} samples')

    print('Converting datasets to sklearn format and training PCA + Random Forest classifier...')
    # Train a simple PCA + Random Forest classifier
    model_skl = make_pca_rf_pipeline(n_pca_comp=150, n_trees=100, random_state=42)

    X_train, y_train = convert_dataset_sklearn(train_dataset)
    X_test, y_test = convert_dataset_sklearn(test_dataset)
    print('Shape of test data:', X_test.shape, y_test.shape)
    model_skl.fit(X_train, y_train)
    test_accuracy = model_skl.score(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    pred_labels = model_skl.predict(X_test[0:16])
    images = [test_dataset[i][0] for i in range(16)]
    show_images_with_predictions(images, y_test[0:16], pred_labels)

    print('Done.')
