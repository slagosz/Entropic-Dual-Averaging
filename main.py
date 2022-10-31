import os

import cascaded_tanks_system.experiment_different_stepsizes, cascaded_tanks_system.experiment_err_vs_n, \
    cascaded_tanks_system.experiment_extra_noise, cascaded_tanks_system.experiment_more_epochs, \
    cascaded_tanks_system.experiment_time_of_estimation, cascaded_tanks_system.search_best_parameters_aggr, \
    cascaded_tanks_system.search_best_parameters_aggr_cv, cascaded_tanks_system.search_best_parameters_da, \
    cascaded_tanks_system.search_best_parameters_da_cv

import coupled_electrical_drives.experiment_err_vs_n, coupled_electrical_drives.experiment_extra_noise, \
    coupled_electrical_drives.search_best_parameters_aggr, coupled_electrical_drives.search_best_parameters_aggr_cv, \
    coupled_electrical_drives.search_best_parameters_da, coupled_electrical_drives.search_best_parameters_da_cv

import wiener_hammerstein.experiment_err_vs_n, wiener_hammerstein.experiment_extra_noise, \
    wiener_hammerstein.search_best_parameters_aggr, wiener_hammerstein.search_best_parameters_aggr_cv, \
    wiener_hammerstein.search_best_parameters_da, wiener_hammerstein.search_best_parameters_da_cv


def delete_results():
    dirs = ["./cascaded_tanks_system/results", "./coupled_electrical_drives/results", "./wiener_hammerstein/results"]
    for dir in dirs:
        dir = os.path.join(os.path.dirname(__file__), dir)
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))


options = [
    ("Quit", exit),
    ("Delete partial results\n", delete_results),

    ("Cascaded tanks system: grid search for the EDA algorithm", cascaded_tanks_system.search_best_parameters_da.run),
    ("Cascaded tanks system: grid search for the EDA algorithm (cross-validation)",
     cascaded_tanks_system.search_best_parameters_da_cv.run),
    ("Cascaded tanks system: grid search for the l1 aggregation algorithm",
     cascaded_tanks_system.search_best_parameters_aggr.run),
    ("Cascaded tanks system: grid search for the l1 aggregation algorithm (cross-validation)",
     cascaded_tanks_system.search_best_parameters_aggr_cv.run),
    ("Cascaded tanks system: different stepsizes", cascaded_tanks_system.experiment_different_stepsizes.run),
    ("Cascaded tanks system: error vs number of measurements", cascaded_tanks_system.experiment_err_vs_n.run),
    ("Cascaded tanks system: time of estimation", cascaded_tanks_system.experiment_time_of_estimation.run),
    ("Cascaded tanks system: error vs variance of extra noise", cascaded_tanks_system.experiment_extra_noise.run),
    ("Cascaded tanks system: error vs number of epochs\n", cascaded_tanks_system.experiment_more_epochs.run),

    ("Coupled electrical drives: grid search for the EDA algorithm", coupled_electrical_drives.search_best_parameters_da.run),
    ("Coupled electrical drives: grid search for the EDA algorithm (cross-validation)",
     coupled_electrical_drives.search_best_parameters_da_cv.run),
    ("Coupled electrical drives: grid search for the l1 aggregation algorithm",
     coupled_electrical_drives.search_best_parameters_aggr.run),
    ("Coupled electrical drives: grid search for the l1 aggregation algorithm (cross-validation)",
     coupled_electrical_drives.search_best_parameters_aggr_cv.run),
    ("Coupled electrical drives: error vs number of measurements", coupled_electrical_drives.experiment_err_vs_n.run),
    ("Coupled electrical drives: error vs variance of extra noise\n", coupled_electrical_drives.experiment_extra_noise.run),

    ("Wiener-Hammerstein circuit: grid search for the EDA algorithm",
     wiener_hammerstein.search_best_parameters_da.run),
    ("Wiener-Hammerstein circuit: grid search for the EDA algorithm (cross-validation)",
     wiener_hammerstein.search_best_parameters_da_cv.run),
    ("Wiener-Hammerstein circuit: grid search for the l1 aggregation algorithm",
     wiener_hammerstein.search_best_parameters_aggr.run),
    ("Wiener-Hammerstein circuit: grid search for the l1 aggregation algorithm (cross-validation)",
     wiener_hammerstein.search_best_parameters_aggr_cv.run),
    ("Wiener-Hammerstein circuit: error vs number of measurements", wiener_hammerstein.experiment_err_vs_n.run),
    ("Wiener-Hammerstein circuit: error vs variance of extra noise",
     wiener_hammerstein.experiment_extra_noise.run)
]

options_dict = {i: option for i, option in enumerate(options)}

options_str = "=================================================================\n"
for key, val in options_dict.items():
    options_str += f"{key} : {val[0]}\n"
options_str += "=================================================================\n"

while True:
    print(options_str)
    try:
        option = int(input())
    except ValueError:
        print("Invalid input!")
        continue

    if option not in options_dict.keys():
        print(f"Option {option} not recognized!")
    else:
        options[option][1]()
