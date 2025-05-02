from binary_classifier import evolve_binary_classifier
from params import *
from windows import window_3x3
from multiprocessing import Process

def evolve_bin_classifiers(batch_size=8, batch_count=4, noises=noise_types):
    for noise_type in noises:
        for i in range(batch_count):
            print(f"Starting batch {i+1}/{batch_count}, {noise_type}")
            runs = []
            for i in range(batch_size):
                seed = get_new_seed()
                p = Process(target=evolve_binary_classifier, args=(noise_type, window_3x3, seed,))
                p.start()
                runs.append(p)
            for p in runs:
                p.join()

evolve_bin_classifiers(batch_size=6, batch_count=2 , noises=["vertical_noise"])