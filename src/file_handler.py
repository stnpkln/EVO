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

# Funkce pro načtení datasetu
def get_dataset():
    if os.path.exists(f"{dataset_dir}/{dataset_file}"):
        if print_dataset_info:
            print("🔄 Načítání datasetu z disku...")
        with open(f"{dataset_dir}/{dataset_file}", "rb") as f:
            dataset = pickle.load(f)
    else:
        raise ValueError("Dataset not found. Please run the script to create the dataset (/preprocessing.ipynb).")
    return dataset

def get_train_data_filter(noise_type):
    dataset = get_dataset()
    train_dataset = []
    for img in testing_set_imgs:
        training_data = []
        training_data.append((dataset[img][noise_type], dataset[img]["original"]))

    return train_dataset

def get_test_data_filter(noise_type):
    dataset = get_dataset()
    test_dataset = []
    for img in testing_set_imgs:
        test_data = []
        test_data.append((dataset[img][noise_type], dataset[img]["original"]))

    return test_dataset

def get_train_data_classifier(noise_type):
    dataset = get_dataset()
    train_dataset = []
    for img in training_set_imgs:
        train_dataset.append((dataset[img][noise_type], dataset[img][f"{noise_type}_mask"]))

    return train_dataset

def get_test_data_classifier(noise_type):
    dataset = get_dataset()
    test_dataset = []
    for img in testing_set_imgs:
        test_dataset.append((dataset[img][noise_type], dataset[img][f"{noise_type}_mask"]))

    return test_dataset

def save_results(results, best_individual, results_filename, classifier_filename):
    # Save the results to a file
    with open(f"{result_dir}/{results_filename}", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {result_dir}/{results_filename}")

    # Save the classifier to a file
    with open(f"{result_dir}/{classifier_filename}", "wb") as f:
        pickle.dump(best_individual, f)
    print(f"Classifier saved to {result_dir}/{classifier_filename}")

    # save summary
    with open(f"{result_dir}/{classifier_stats}.csv", "a") as f:
        f.write(f"{seed},{mutation_rate},{max_evaluations},{results['accuracy']},{results['precision']},{results['recall']}\n")



