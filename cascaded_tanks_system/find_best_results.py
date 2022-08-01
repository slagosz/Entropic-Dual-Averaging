import os
import pickle

def print_best_results(path, algorithm_name):
    files = [file for file in os.listdir(path) if file.endswith('.pz') and algorithm_name in file]

    all_results = []
    for file in files:
        fp = os.path.join(path, file)
        with open(fp, 'rb') as f:
            results = pickle.load(f)

        all_results.append(results)

    all_results = sorted(all_results, key=lambda r: r['avg_error'])

    for results in all_results[0:10]:
        print(results)


print_best_results('results', 'EntropicDualAveragingAlgorithm')
