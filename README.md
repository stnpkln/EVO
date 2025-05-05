# Evolving image filter using CGP
Project for EVO, evolves a binary classifier (classifies pixel damage) and an image filter using this classifier. 
Experiments mainly concern different window types (or "kernels").

- scripts are supposed to be run from the root directory of the repo.

## Setup
- Create venv in `\` using `requirements.txt` for processing in python notebooks
- Create venv in `\src` using `src\requirements.txt` for running cgp (cartesian genetic programming)
- required python version is python 3.8.20 (because of the hal-cgp library)

## Running
- create the datasets using `preprocessing.ipynb`
- run experiments in `\src`
- Visualize results in `result_processing.ipynb`

# Time
- classifier: 2000 eval. / 3 min = 666/minute (8 processes)
- image filter: 5000 eval. / 2 min = 2500/minute (1 process)

# TODO 
- also evolve salt and pepper
- compare with median filter