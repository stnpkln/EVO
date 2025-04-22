import matplotlib.pyplot as plt
import numpy as np
from binary_classifier import get_evolved_classifier
from evaluation import evaluate_filter
from file_handler import get_train_data_filter, save_filter, save_filter_results
from params import *
from windows import apply_window, window_3x3
import cgp

class Filter:
    def __init__(self, filter_func, classifier, window):
        self.classifier = classifier
        self.window = window
        self.filter_func = filter_func

    def __call__(self, image):
        classified_image = self.classifier(image)
        for x in range(len(image)):
            for y in range(len(image[0])):
                if classified_image[x][y] != 255:
                    continue
                filter_input = apply_window(image, self.window, x, y)
                prediction = self.filter_func(*filter_input)
                image[x][y] = prediction

        return image

# turn off numpy overflow warnings (overflows are expected)
np.seterr(over='ignore', under='ignore')
############################################################
## EVOLVE
############################################################
def evolve_image_filter(classifier_seed, noise_type, window, seed):
    random.seed(seed)
    if (noise_type not in noise_types):
        raise ValueError(f"Noise type {noise_type} is not supported. Supported noise types are: {noise_types}")
    
    result_file_name = f"results_filter_{noise_type}_{window['name']}_{seed}.pkl"
    filter_file_name = f"best_filter_{seed}.pkl"

    ##############################################
    ## LOAD DATASET AND CLASSIFIER
    ##############################################
    
    training_data = get_train_data_filter(noise_type)
    classifier = get_evolved_classifier(classifier_seed)
    
    ##############################################
    ## CLASSIFY PIXEL DAMAGE 
    ##############################################
    classified_images = []
    for (noised_image, _) in training_data:
        classified_images.append(classifier(noised_image))
        

    damaged_pixel_coords_per_image = []
    for classified_image in classified_images:
        damaged_pixel_coords = []
        for x in range(len(noised_image)):
            for y in range(len(noised_image[0])):
                if (classified_image[x][y] == 255):
                    damaged_pixel_coords.append((x, y))
        damaged_pixel_coords_per_image.append(damaged_pixel_coords)

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
        filter = individual.to_func()
        for i in range(len(training_data)):
            noised_image, original = training_data[i]
            damaged_pixels = damaged_pixel_coords_per_image[i]
            for (x, y) in damaged_pixels:
                filter_input = apply_window(noised_image, window, x, y)
                prediction = filter(*filter_input)
                sse += (float(prediction) - float(original[x, y])) ** 2
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

    # TODO
    # evaluate result on testing dataset
    best_individual = results["best_individual"]
    filter_func = best_individual.to_func()
    filter = Filter(filter_func, classifier, window)
    filtered_image, original_image, noised_image = evaluate_filter(filter, noise_type)

    # save the results and config
    results["seed"] = seed
    results["noise"] = noise_type
    results["window"] = window["name"]
    results["filtered_image"] = filtered_image
    results["original_image"] = original_image
    results["noised_image"] = noised_image
    results["classifier_seed"] = classifier_seed
    results["config"] = get_config()
    del results["best_individual"] # remove the individual from the results, it is saved separately, cgp library causes problems

    save_filter_results(results, result_file_name)
    save_filter(best_individual, window, classifier_seed, filter_file_name)
    return results

evolve_image_filter(classifier_seed="1563348194",
                    noise_type=noise_types[0],
                    window=window_3x3,
                    seed=seed)