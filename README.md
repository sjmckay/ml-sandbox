# ml-sandbox
Load galaxy images and test ML architectures for classifying and fitting models.

The dataset used here is from the Galaxy Zoo (GZ), in which citizen scientists help to label galaxies (e.g., is is a merger, is it isolated, are there tidal tail features visible, is it a disk, is it a spheroid, etc). The question posed in this project is whether we can train a robust ML model to classify galaxy images (RGB) based on the GZ data.

Current setup: 
- implemented robust CNN and PCA+RF classifiers on example morphological datasets, e.g., merging galaxies, from CANDELS fields.
- compared performance between methods.
- currently, classes are assigned based on maximum vote and the weights are assigned based on the number of members of each class (because galaxy samples are highly imbalanced datasets), but an improvement would be to *also* weight them by the agreement of the human votes (i.e., don't weight a galaxy that had 5 votes for merger and 4 votes for isolated the same as one that was 9 votes for merger).
- Future: test transformer architecture on classification of various features.

Files:
- `models.py`: model classes for classification
- `dataset.py`: handles galaxy dataset loading for PyTorch 
- `train_eval.py`: handles model training and validation
- `utils.py`: common util functions

See `ex_flow_torch.py` and `ex_flow_skl.py` for example usage with PyTorch and Sklearn classification of galaxy mergers. 


