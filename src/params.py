import random
from datetime import datetime, timezone
from operations import *
from time import sleep
from windows import window_3x3, window_for_vertical, window_for_diagonal

def get_new_seed():
    global seed
    sleep(0.01)
    seed = int(datetime.now(timezone.utc).timestamp() * 1000) % (2 ** 32)
    return seed
##############################################
## DATASET PARAMETERS
##############################################
# WARNING: If you change the sample size, image width or image height, you need to delete the dataset files first!
print_dataset_info = False # Print information about the dataset
noise_types = ["vertical_noise", "diagonal_noise", "periodic_noise"]
window_types = ["window_3x3", "window_for_vertical", "window_for_diagonal"]
windows_per_noise = {
    "vertical_noise": [window_3x3, window_for_vertical],
    "diagonal_noise": [window_3x3, window_for_diagonal],
    "periodic_noise": [window_3x3, window_for_vertical]
}
classifiers_seeds_per_noise = {
    "vertical_noise": "2291659237",
    "diagonal_noise": "2441695047",
    "periodic_noise": "2440749822"
}
training_set_imgs = ["moon", "astronaut"]
testing_set_imgs = ["camera"]
image_width = 128 # Width of the image (14 pixels)
image_height = 128 # Height of the image (14 pixels)
dataset_dir = "datasets"
dataset_file = "dataset.pkl"
result_dir = "results"
test_image = "camera"
classifier_stats = "classifier_stats"
filter_stats = "filter_stats"
classifier_threshold = 127

##############################################
## CGP PARAMETERS
##############################################
print_evo_progress = True # Print progress overall classifier
n_columns = 60 # Number of columns in the CGP
n_rows = 1 # Number of rows in the CGP
n_levels_back = n_columns # Number of levels back in the CGP
primitives = ( 
    Constant255,
    Identity,
    Inversion,
    Max,
    Min,
    UINTDivBy2,
    UINTDivBy4,
    UINT8Add,
    UINT8AddSat,
    Average,
    ConditionalAssign,
    AbsoluteDiff
) # List of primitives to use in the CGP
n_inputs = 9 # Number of inputs, !! DEPENDS ON CHOSEN WINDOW !!
n_outputs = 1 # Number of outputs (1 for binary classification)
n_parents = 1 # Number of parents to select for breeding
n_offsprings = 4 # Number of offsprings to generate
mutation_rate = 0.1 # Mutation rate
termination_fitness = 0.0 # Termination fitness
max_evaluations = 1e5 # Maximum number of fitness evaluations
max_generations = max_evaluations / (n_parents + n_offsprings) # Maximum number of generations
n_processes = 1 # Number of processes for parallel evaluation

def get_config():
	return {
        "image_width": image_width,
        "image_height": image_height,
        "dataset_dir": dataset_dir,
        "result_dir": result_dir,
        "n_columns": n_columns,
        "n_rows": n_rows,
        "n_levels_back": n_levels_back,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "n_parents": n_parents,
        "n_offsprings": n_offsprings,
        "mutation_rate": mutation_rate,
        "termination_fitness": termination_fitness,
        "max_evaluations": max_evaluations,
        "max_generations": max_generations,
        "n_processes": n_processes,
		"classifier_threshold": classifier_threshold,
    }