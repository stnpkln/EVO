import random
from datetime import datetime, timezone
from operations import *

seed = int(datetime.now(timezone.utc).timestamp() * 1000) % (2 ** 32) # Seed for reproducibility
##############################################
## DATASET PARAMETERS
##############################################
# WARNING: If you change the sample size, image width or image height, you need to delete the dataset files first!
print_dataset_info = False # Print information about the dataset
noise_types = ["vertical_noise", "diagonal_noise", "periodic_noise"]
training_set_imgs = ["moon", "astronaut"]
testing_set_imgs = ["camera"]
image_width = 128 # Width of the image (14 pixels)
image_height = 128 # Height of the image (14 pixels)
dataset_dir = "datasets"
result_dir = "results"
test_image = "camera"

##############################################
## CGP PARAMETERS
##############################################
print_progress = True # Print progress overall classifier
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
n_inputs = image_width * image_height # Number of inputs (flattened image)
n_outputs = 1 # Number of outputs (1 for binary classification)
n_parents = 1 # Number of parents to select for breeding
n_offsprings = 4 # Number of offsprings to generate
mutation_rate = 0.1 # Mutation rate
termination_fitness = 0.0 # Termination fitness
max_evaluations = 1e2 # Maximum number of generations
max_generations = max_evaluations / (n_parents + n_offsprings) # Maximum number of generations
n_processes = 1 # Number of processes for parallel evaluation

def get_config():
	return {
		"seed": seed,
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
    }