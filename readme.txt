Evolving image filter using CGP
Project for EVO, evolves a binary classifier (classifies pixel damage) and an image filter using this classifier. 
Experiments mainly concern different window types (or "kernels").

- scripts are supposed to be run from the root directory of the repo.

Setup
- Create venv in `\` using `requirements.txt` for processing in python notebooks
- Create venv in `\src` using `src\requirements.txt` for running cgp (cartesian genetic programming)
- required python version is python 3.8.20 (because of the hal-cgp library)

Running
- most of global configuration can be set in params.py
- run experiments in `\src` The recommended order is:
	- create datasets with `preprocessing.ipynb`
	- create pixel damage classifiers with `classifier_experiments.py`
	- create image filters with `image_filter_experiments.py`
	- Visualize results with `result_processing.py`

