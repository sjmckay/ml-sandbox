# ml-sandbox
Load galaxy images and test ML architectures for classifying and fitting models.

Current plans: 
- implement robust CNN and PCA+RF classifiers on example morphological datasets from CANDELS fields.
- compare performance to new data from **other** fields.
- test various transformer architectures on classification of various features.

Files:
- `models.py`: various model classes for classification
- `dataset.py`: handles galaxy dataset loading for PyTorch 
- `train_eval.py`: handles model training and validation
- `utils.py`: common util functions

See `sandbox.ipynb` notebook for some example usage and testing. 

Hopefully will build out `example_workflow.py` as well for a complete script.


