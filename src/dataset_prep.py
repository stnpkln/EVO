import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import pickle
from PIL import Image
from params import *

######################################################
## LOAD AND PREPARE DATASET
######################################################

# Název souboru pro uložení datasetu
DATASET_FILE = "datasets.pkl"

# Funkce pro načtení datasetu
def get_dataset():
    if os.path.exists(f"{dataset_dir}/{DATASET_FILE}"):
        if print_dataset_info:
            print("🔄 Načítání datasetu z disku...")
        with open(f"{dataset_dir}/{DATASET_FILE}", "rb") as f:
            dataset = pickle.load(f)
    else:
        raise ValueError("Dataset not found. Please run the script to create the dataset (/preprocessing.ipynb).")
    return dataset

def get_train_data(noise_type):
    dataset = get_dataset()
    train_dataset = []
    for img in testing_set_imgs:
        training_data = []
        training_data.append((dataset[img][noise_type], dataset[img]["original"]))

    return train_dataset

def get_test_data(noise_type):
    dataset = get_dataset()
    test_dataset = []
    for img in testing_set_imgs:
        test_data = []
        test_data.append((dataset[img][noise_type], dataset[img]["original"]))

    return test_dataset





