# ml-sandbox
Supervised machine learning workflows for classifying galaxy images, based on Galaxy Zoo data. Currently supports traditional ML (PCA and Random Forest) and deep learning methods (CNN).

The dataset used here is from the Galaxy Zoo (GZ), in which citizen scientists help to label galaxies (e.g., is is a merger, is it isolated, are there tidal tail features visible, is it a disk, is it a spheroid, etc). The question posed in this project is whether we can train a robust ML model to classify galaxy images (RGB) based on the GZ data.

Current setup: 
- implemented CNN and PCA+RF classifier workflows on example morphological datasets, e.g., merging galaxies, from CANDELS fields.
- designed visualizations and evaluation metrics to compare performance between methods.


Todos: 
- classes are assigned based on maximum vote and the weights are assigned based on the number of members of each class (because galaxy samples are highly imbalanced datasets), but an improvement would be to *also* weight them by the agreement of the human votes (i.e., don't weight a galaxy that had 5 votes for merger and 4 votes for isolated the same as one that was 9 votes for merger).
- test transformer architecture on classification of various features.

Files:
- `models.py`: model classes (PyTorch and Scikit-learn) for classification
- `dataset.py`: galaxy dataset loading for PyTorch 
- `train_eval.py`: model training and validation
- `utils.py`: common util functions (Todo: expand)

See `ex_flow_torch.py` and `ex_flow_skl.py` for example usage with PyTorch and Sklearn classification of galaxy mergers. 


