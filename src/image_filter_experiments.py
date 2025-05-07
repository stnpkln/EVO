from params import *
from multiprocessing import Process
from image_filter import evolve_image_filter

def evolve_image_filters(batch_size=8, batch_count=4, noises=noise_types):
    for noise_type in noises:
        for window in windows_per_noise[noise_type]:
            for i in range(batch_count):
                print(f"Starting batch {i+1}/{batch_count}, {noise_type}, {window['name']}")
                runs = []
                for i in range(batch_size):
                    seed = get_new_seed()
                    p = Process(target=evolve_image_filter, args=(classifiers_seeds_per_noise[noise_type], noise_type, window, seed,))
                    p.start()
                    runs.append(p)
                for p in runs:
                    p.join()

evolve_image_filters(batch_size=8, batch_count=4, noises=["periodic_noise"])
