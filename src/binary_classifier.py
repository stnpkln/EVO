import matplotlib.pyplot as plt
import numpy as np
from dataset_prep import get_train_data
from params import *
import cgp

# turn off numpy overflow warnings (overflows are expected)
np.seterr(over='ignore', under='ignore')
############################################################
## EVOLVE
############################################################
def get_binary_classifier(noise_type, seed):
    random.seed(seed)
    if (noise_type not in noise_types):
        raise ValueError(f"Noise type {noise_type} is not supported. Supported noise types are: {noise_types}")

    ##############################################
    ## LOAD DATASET
    ##############################################
    
    (X_train1, y_train1), (X_train2, y_train2) = get_train_data(noise_type)

    # convert to binary classification - if the label is the number to classify, set it to 255, otherwise set it to 0

    ##############################################
    ## DEFINE OBJECTIVE FUNCTION
    ##############################################

    def objective(individual):
        if not individual.fitness_is_None():
            return individual

        mse = 0.0
        classify = individual.to_func()
        # TODO somehow do the filter shapes
        # for image, label in zip(X_train, y_train):
        #     # Reshape the image to match the input shape of the individual
        #     prediction = classify(*image)
        #     sum_abs_diff += abs(float(prediction) - float(label))
        # individual.fitness = -sum_abs_diff / training_size
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
                    print_progress=print_single_progress,
                    callback=recording_callback
    )

    # save the results to a file
    results["seed"] = seed
    results["config"] = get_config()
    save_results(results, result_file_name)
    return results