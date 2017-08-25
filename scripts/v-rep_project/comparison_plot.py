import os
import time
# import json
# import numpy as np
from matplotlib import pyplot as plt
from evaluate_curriculum import Curriculum
from robotenv import RobotEnv

# exp_with_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE
# exp_without_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE

current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_curriculums_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
# exp_with_cl_dir_path = os.path.join(all_models_dir_path, exp_with_cl_dir_name)
# exp_without_cl_dir_path = os.path.join(all_models_dir_path, exp_without_cl_dir_name)
comparison_plots_dir_path = os.path.join(all_curriculums_dir_path, "comparison_plots")
os.makedirs(comparison_plots_dir_path, exist_ok=True)

# lists_to_deserialize = ['cl_switching_eps', 'cl_cumul_successes_list', 'cl_switching_steps', 'cl_test_success_rates_list', 'cl_test_steps_list']  # have to be lists
# for list_to_deserialize in lists_to_deserialize:
#     list_json_file = os.path.join(experiment_dir_path, list_to_deserialize + '.json')
#     with open(list_json_file, "r") as json_file:
#         # json.dump(eval(list_to_serialize), json_file)

# data  = json.loads(array)

#     fig = plt.figure()
#     plt.plot(test_step_numbers, test_success_rate, linewidth=0.5)
#     plt.ylabel('success rate')
#     plt.xlabel('steps')
#     plt.title('Success rate in test conditions')
#     # vertical lines at subtask switching
#     for switch_ep in subtask_switch_steps:
#         plt.axvline(x=switch_ep, ls='dashed', color='r')
#     plot_file = os.path.join(dir_path, 'success_rate.svg')
#     fig.savefig(plot_file, bbox_inches='tight')
#     plt.close()


def savePlot(dir_path,
             x_label, no_curr_x_values, curr_x_values,
             y_label, no_curr_y_values, curr_y_values,
             title,
             filename,
             vertical_xs=None
             ):
    fig = plt.figure()
    plt.plot(curr_x_values, curr_y_values, 'b', linewidth=0.5, label='with curriculum')
    plt.plot(no_curr_x_values, no_curr_y_values, 'r--', linewidth=0.5, label='no curriculum')
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if vertical_xs is not None:
        # vertical lines at subtask switching
        for vertical_x in vertical_xs:
            plt.axvline(x=vertical_x, color='g', ls=':', linewidth=0.5)
    plot_file = os.path.join(dir_path, filename + '.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()


# ###################
no_curriculum = Curriculum.NO_CURRICULUM_VEL_025

Curriculums = [Curriculum.CURRICULUM_INITIALIZE_FURTHER,
               Curriculum.CURRICULUM_DECREASING_SPEED,
               Curriculum.CURRICULUM_INCREASING_JOINT_NUMBER
               ]

task = RobotEnv.TASK_PUSH_CUBE_TO_TARGET_POSITION

testing_scripts = False
max_steps_per_episode = 200
num_episodes = 1000
num_hidden_layers = 3
num_neurons_per_hidden = 50
max_updates_per_env_step = 10
batch_size = 32
lrate = 1e-4
replay_start_size = (num_episodes // 20) * max_steps_per_episode
replay_memory_size = 10 * replay_start_size
disable_saving = True

# ################

curr_args = dict(curriculum=no_curriculum,
                 task=task,
                 max_steps_per_episode=max_steps_per_episode,
                 num_episodes=num_episodes,
                 num_hidden_layers=num_hidden_layers,
                 num_neurons_per_hidden=num_neurons_per_hidden,
                 batch_size=batch_size,
                 lrate=lrate,
                 testing_scripts=testing_scripts,  # ##
                 max_updates_per_env_step=max_updates_per_env_step,
                 replay_start_size=replay_start_size,
                 replay_memory_size=replay_memory_size,
                 disable_saving=disable_saving,
                 )

curr_args.update(dict(curriculum=no_curriculum))
no_curr = Curriculum(**curr_args)
no_curriculum_results_dict = no_curr.run()

no_curriculum_undisc_return_per_ep = no_curriculum_results_dict['curriculum_undisc_return_per_ep']
no_curriculum_num_steps_per_ep = no_curriculum_results_dict['curriculum_num_steps_per_ep']
no_curriculum_cumul_successes_per_ep = no_curriculum_results_dict['curriculum_cumul_successes_per_ep']
no_curriculum_epsilon_per_ep = no_curriculum_results_dict['curriculum_epsilon_per_ep']
no_curriculum_success_step_per_ep = no_curriculum_results_dict['curriculum_success_step_per_ep']
no_curriculum_test_steps = no_curriculum_results_dict['curriculum_test_steps']
no_curriculum_test_episodes = no_curriculum_results_dict['curriculum_test_episodes']
no_curriculum_test_success_rates = no_curriculum_results_dict['curriculum_test_success_rates']
no_curriculum_test_mean_returns = no_curriculum_results_dict['curriculum_test_mean_returns']
no_curriculum_net_updates_per_step = no_curriculum_results_dict['curriculum_net_updates_per_step']
# no_curriculum_switching_episodes = no_curriculum_results_dict['curriculum_switching_episodes']
# no_curriculum_switching_steps = no_curriculum_results_dict['curriculum_switching_steps']

no_curriculum_episodes = range(1, len(no_curriculum_undisc_return_per_ep) + 1)
no_curriculum_steps = range(1, len(no_curriculum_net_updates_per_step) + 1)

no_curriculum_folder_name = no_curr.folder_name

# ####

for curriculum in Curriculums:
    curr_args.update(dict(curriculum=curriculum))
    curr = Curriculum(**curr_args)
    results_dict = curr.run()

    curriculum_undisc_return_per_ep = results_dict['curriculum_undisc_return_per_ep']
    curriculum_num_steps_per_ep = results_dict['curriculum_num_steps_per_ep']
    curriculum_cumul_successes_per_ep = results_dict['curriculum_cumul_successes_per_ep']
    curriculum_epsilon_per_ep = results_dict['curriculum_epsilon_per_ep']
    curriculum_success_step_per_ep = results_dict['curriculum_success_step_per_ep']
    curriculum_test_steps = results_dict['curriculum_test_steps']
    curriculum_test_episodes = results_dict['curriculum_test_episodes']
    curriculum_test_success_rates = results_dict['curriculum_test_success_rates']
    curriculum_test_mean_returns = results_dict['curriculum_test_mean_returns']
    curriculum_net_updates_per_step = results_dict['curriculum_net_updates_per_step']
    curriculum_switching_episodes = results_dict['curriculum_switching_episodes']
    curriculum_switching_steps = results_dict['curriculum_switching_steps']

    curriculum_episodes = range(1, len(curriculum_undisc_return_per_ep) + 1)
    curriculum_steps = range(1, len(curriculum_net_updates_per_step) + 1)

    curriculum_folder_name = curr.folder_name

    #  ###
    # curr_args.update(dict(curriculum=no_curriculum))
    # no_curr = Curriculum(**curr_args)
    # no_curriculum_results_dict = no_curr.run()

    # no_curriculum_undisc_return_per_ep = no_curriculum_results_dict['curriculum_undisc_return_per_ep']
    # no_curriculum_num_steps_per_ep = no_curriculum_results_dict['curriculum_num_steps_per_ep']
    # no_curriculum_cumul_successes_per_ep = no_curriculum_results_dict['curriculum_cumul_successes_per_ep']
    # no_curriculum_epsilon_per_ep = no_curriculum_results_dict['curriculum_epsilon_per_ep']
    # no_curriculum_success_step_per_ep = no_curriculum_results_dict['curriculum_success_step_per_ep']
    # no_curriculum_test_steps = no_curriculum_results_dict['curriculum_test_steps']
    # no_curriculum_test_episodes = no_curriculum_results_dict['curriculum_test_episodes']
    # no_curriculum_test_success_rates = no_curriculum_results_dict['curriculum_test_success_rates']
    # no_curriculum_test_mean_returns = no_curriculum_results_dict['curriculum_test_mean_returns']
    # no_curriculum_net_updates_per_step = no_curriculum_results_dict['curriculum_net_updates_per_step']
    # # no_curriculum_switching_episodes = no_curriculum_results_dict['curriculum_switching_episodes']
    # # no_curriculum_switching_steps = no_curriculum_results_dict['curriculum_switching_steps']

    # no_curriculum_episodes = range(1, len(no_curriculum_undisc_return_per_ep) + 1)
    # no_curriculum_steps = range(1, len(no_curriculum_net_updates_per_step) + 1)

    # no_curriculum_name = no_curr.curriculum_name
    # ###

    timestr = time.strftime("%Y-%b-%d_%H-%M-%S", time.gmtime())  # or time.localtime()
    current_comparison_name = timestr + '_comparison_' + no_curriculum_folder_name + '_' + curriculum_folder_name
    current_comparison_dir_path = os.path.join(comparison_plots_dir_path, current_comparison_name)
    os.makedirs(current_comparison_dir_path, exist_ok=True)

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_episodes, curriculum_episodes,
             'returns', no_curriculum_undisc_return_per_ep, curriculum_undisc_return_per_ep,
             'Undiscounted returns during training',
             'comparison_undisc_return_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_episodes, curriculum_episodes,
             'steps', no_curriculum_num_steps_per_ep, curriculum_num_steps_per_ep,
             'Environment steps',
             'curriculum_num_steps_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_episodes, curriculum_episodes,
             'successful episodes', no_curriculum_cumul_successes_per_ep, curriculum_cumul_successes_per_ep,
             'Cumulative successful episodes',
             'curriculum_cumul_successes_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_episodes, curriculum_episodes,
             'epsilon', no_curriculum_epsilon_per_ep, curriculum_epsilon_per_ep,
             'Epsilon evolution',
             'curriculum_epsilon_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_episodes, curriculum_episodes,
             'steps', no_curriculum_success_step_per_ep, curriculum_success_step_per_ep,
             'Steps to reach a goal state',
             'curriculum_success_step_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'steps', no_curriculum_test_steps, curriculum_test_steps,
             'success rate', no_curriculum_test_success_rates, curriculum_test_success_rates,
             'Success rate in test conditions',
             'curriculum_success_rate_per_step',
             curriculum_switching_steps,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_test_episodes, curriculum_test_episodes,
             'success rate', no_curriculum_test_success_rates, curriculum_test_success_rates,
             'Success rate in test conditions',
             'curriculum_success_rate_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'steps', no_curriculum_test_steps, curriculum_test_steps,
             'mean return', no_curriculum_test_mean_returns, curriculum_test_mean_returns,
             'Mean undisc. return in test conditions',
             'curriculum_test_mean_returns_per_step',
             curriculum_switching_steps,
             )

    savePlot(current_comparison_dir_path,
             'episodes', no_curriculum_test_episodes, curriculum_test_episodes,
             'mean return', no_curriculum_test_mean_returns, curriculum_test_mean_returns,
             'Mean undisc. return in test conditions',
             'curriculum_test_mean_returns_per_ep',
             curriculum_switching_episodes,
             )

    savePlot(current_comparison_dir_path,
             'steps', no_curriculum_steps, curriculum_steps,
             'updates', no_curriculum_net_updates_per_step, curriculum_net_updates_per_step,
             'Number of network updates',
             'curriculum_net_updates_per_step',
             curriculum_switching_steps,
             )
