import matplotlib.pyplot as plt
import numpy as np
from file_handler import get_train_data_classifier, save_results
from params import *
from windows import apply_window, window_3x3
import cgp

# turn off numpy overflow warnings (overflows are expected)
np.seterr(over='ignore', under='ignore')
############################################################
## EVOLVE
############################################################
def get_binary_classifier(noise_type, window, seed):
    random.seed(seed)
    if (noise_type not in noise_types):
        raise ValueError(f"Noise type {noise_type} is not supported. Supported noise types are: {noise_types}")
    
    result_file_name = f"results_{noise_type}_{window['name']}_{seed}.pkl"

    ##############################################
    ## LOAD DATASET
    ##############################################
    
    training_data = get_train_data_classifier(noise_type)

    # convert to binary classification - if the label is the number to classify, set it to 255, otherwise set it to 0

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
        print(len(training_data))
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

    n_inputs = image_width * image_height
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

    # save the results to a file
    results["seed"] = seed
    results["noise"] = noise_type
    results["window"] = window["name"]
    results["config"] = get_config()

    save_results(results, result_file_name)
    return results

get_binary_classifier(noise_types[0], window=window_3x3, seed=seed)