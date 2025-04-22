import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from evaluation import evaluate_classifier
from file_handler import get_train_data_classifier, save_classifier, save_classifier_results
from params import *
from windows import apply_window, window_3x3
import cgp

class Classifier:
    def __init__(self, classifier_func, window, threshold):
        self.classifier_func = classifier_func
        self.window = window
        self.threshold = threshold

    def __call__(self, image):
        y_pred = []
        for x in range(len(image)):
            for y in range(len(image[0])):
                classifier_input = apply_window(image, self.window, x, y)
                prediction = self.classifier_func(*classifier_input)
                classification = 0
                if prediction > self.threshold:
                    classification = 255
                y_pred.append(classification)
        return np.array(y_pred).reshape(image.shape)

# turn off numpy overflow warnings (overflows are expected)
np.seterr(over='ignore', under='ignore')
############################################################
## EVOLVE
############################################################
def evolve_binary_classifier(noise_type, window, seed):
    random.seed(seed)
    if (noise_type not in noise_types):
        raise ValueError(f"Noise type {noise_type} is not supported. Supported noise types are: {noise_types}")
    
    result_file_name = f"results_classifier_{noise_type}_{window['name']}_{seed}.pkl"
    classifier_file_name = f"best_classifier_{seed}.pkl"

    ##############################################
    ## LOAD DATASET
    ##############################################
    
    training_data = get_train_data_classifier(noise_type)

    ##############################################
    ## SET GLOBAL VARIABLES THAT ARE DEPENDANT
    ##############################################
    n_inputs = len(window["coords"])
    
    ##############################################
    ## DEFINE OBJECTIVE FUNCTION
    ##############################################

    def objective(individual):
        if not individual.fitness_is_None():
            return individual

        sse = 0.0
        predicted_pixels = 0
        classify = individual.to_func()
        # TODO somehow do the filter shapes
        for (noised_image, mask)  in training_data:
            for x in range(len(noised_image)):
                for y in range(len(noised_image[0])):
                    classifier_input = apply_window(noised_image, window, x, y)
                    prediction = classify(*classifier_input)
                    sse += (float(prediction) - float(mask[x, y])) ** 2
                    predicted_pixels += 1

        mse = sse / (predicted_pixels)
        individual.fitness = -mse
        return individual

    ######################################################
    ## DEFINE HISTORY
    ######################################################

    results = {}
    results["best_fitness_hist"] = []
    results["best_individual"] = None

    def recording_callback(pop):
        results["best_fitness_hist"].append(pop.champion.fitness)
        if results["best_individual"] is None or pop.champion.fitness > results["best_individual"].fitness:
            results["best_individual"] = pop.champion

    ########################################################
    ## DEFINE POPULATION
    ########################################################

    genome_params= {
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "n_columns": n_columns,
            "n_rows": n_rows,
            "levels_back": n_levels_back,
            "primitives": primitives,
        }

    population = cgp.Population(
        n_parents=n_parents,
        seed=seed,
        genome_params=genome_params
    )

    #################################################
    ## DEFINE EVOLUTION STRATEGY
    #################################################

    strategy = cgp.ea.MuPlusLambda(
        n_offsprings=n_offsprings,
        mutation_rate=mutation_rate,
        n_processes=n_processes,
        )

    cgp.evolve(pop=population,
                    objective=objective,
                    ea=strategy,
                    termination_fitness=termination_fitness,
                    max_generations=max_generations,
                    print_progress=print_evo_progress,
                    callback=recording_callback
    )


    # evaluate result on testing dataset
    best_individual = results["best_individual"]
    classifier_func = best_individual.to_func()
    classifier = Classifier(classifier_func, window, classifier_threshold)
    accuracy, precision, recall, pred_mask, mask = evaluate_classifier(classifier, noise_type)

    # save the results and config
    results["seed"] = seed
    results["noise"] = noise_type
    results["window"] = window["name"]
    results["accuracy"] = accuracy
    results["precision"] = precision
    results["recall"] = recall
    results["pred_mask"] = pred_mask
    results["mask"] = mask
    results["config"] = get_config()
    del results["best_individual"] # remove the individual from the results, it is saved separately, cgp library causes problems

    save_classifier_results(results, result_file_name)
    save_classifier(best_individual, window, classifier_file_name)
    return results

def get_evolved_classifier(classifier_seed):
    classifier_path = f"{result_dir}/best_classifier_{classifier_seed}.pkl"
    if os.path.exists(classifier_path):
        with open(classifier_path, "rb") as f:
            best_classifier = pickle.load(f)
            classifier_func = best_classifier["best_individual"].to_func()
            classifier = Classifier(classifier_func, best_classifier["window"], best_classifier["threshold"])
    else:
        raise ValueError(f"Classifier {classifier_path} not found.")
    return classifier

# evolve_binary_classifier(noise_types[0], window=window_3x3, seed=seed)