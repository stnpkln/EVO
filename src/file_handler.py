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
        train_dataset.append((dataset[img][noise_type], dataset[img]["original"]))

    return train_dataset

def get_test_data_filter(noise_type):
    dataset = get_dataset()
    test_dataset = []
    for img in testing_set_imgs:
        test_dataset.append((dataset[img][noise_type], dataset[img]["original"]))

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

def save_classifier_results(results, results_filename):
    # Save the results to a file
    with open(f"{result_dir}/{results_filename}", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {result_dir}/{results_filename}")

    # save summary
    with open(f"{result_dir}/{classifier_stats}.csv", "a") as f:
        f.write(f"{results['seed']},{results['noise']},{results['window']},{mutation_rate},{max_evaluations},{results['accuracy']},{results['precision']},{results['recall']}\n")
    print(f"Summary saved to {result_dir}/{classifier_stats}.csv")

def save_filter_results(results, results_filename):
    # Save the results to a file
    with open(f"{result_dir}/{results_filename}", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {result_dir}/{results_filename}")

    # save summary
    with open(f"{result_dir}/{filter_stats}.csv", "a") as f:
        f.write(f"{results['seed']},{results['noise']},{results['window']},{results['classifier_seed']},{mutation_rate},{max_evaluations},{results['best_fitness_hist'].pop(-1)}\n")
    print(f"Summary saved to {result_dir}/{filter_stats}.csv")

def save_classifier(best_individual, window, classifier_filename):
    # Save the classifier to a file
    with open(f"{result_dir}/{classifier_filename}", "wb") as f:
        classifier = {}
        classifier["best_individual"] = best_individual
        classifier["window"] = window
        classifier["threshold"] = classifier_threshold
        pickle.dump(classifier, f)
    print(f"Classifier saved to {result_dir}/{classifier_filename}")

def save_filter(best_individual, window, classifier_seed, filter_filename):
    # Save the filter to a file
    with open(f"{result_dir}/{filter_filename}", "wb") as f:
        filter = {}
        filter["best_individual"] = best_individual
        filter["window"] = window
        filter["classifier_seed"] = classifier_seed
        pickle.dump(filter, f)
    print(f"Filter saved to {result_dir}/{filter_filename}")