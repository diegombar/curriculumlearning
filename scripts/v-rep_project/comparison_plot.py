import os
import json
import numpy as np
from matplotlib import pyplot as plt


exp_with_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE
exp_without_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE

current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
exp_with_cl_dir_path = os.path.join(all_models_dir_path, exp_with_cl_dir_name)
exp_without_cl_dir_path = os.path.join(all_models_dir_path, exp_without_cl_dir_name)

lists_to_deserialize = ['cl_switching_eps', 'cl_cumul_successes_list', 'cl_switching_steps', 'cl_test_success_rates_list', 'cl_test_steps_list']  # have to be lists
for list_to_deserialize in lists_to_deserialize:
    list_json_file = os.path.join(experiment_dir_path, list_to_deserialize + '.json')
    with open(list_json_file, "r") as json_file:
        # json.dump(eval(list_to_serialize), json_file)



dict_with_cl 

dict_without_cl

data  = json.loads(array)




    fig = plt.figure()
    plt.plot(test_step_numbers, test_success_rate, linewidth=0.5)
    plt.ylabel('success rate')
    plt.xlabel('steps')
    plt.title('Success rate in test conditions')
    # vertical lines at subtask switching
    for switch_ep in subtask_switch_steps:
        plt.axvline(x=switch_ep, ls='dashed', color='r')
    plot_file = os.path.join(dir_path, 'success_rate.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()