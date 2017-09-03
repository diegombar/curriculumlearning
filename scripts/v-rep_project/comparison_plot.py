import os
# import time
# import json
import numpy as np
from matplotlib import pyplot as plt
from curriculum import Curriculum
from robotenv import RobotEnv

# exp_with_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE
# exp_without_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE


current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_curriculums_dir_path = os.path.join(current_dir_path, "trained_models_and_results")

exp_no_curr_sparse_folder_name = "Sep-02_02-39-59_graphic04.doc.ic.ac.uk_reaching_no_cl_sparse"  # #########
exp_no_curr_shaping_folder_name = "Sep-01_14-53-22_graphic04.doc.ic.ac.uk_reaching_no_cl_shaping"  # #########

exp_no_curr_sparse_path = os.path.join(all_curriculums_dir_path, exp_no_curr_sparse_folder_name)
exp_no_curr_shaping_path = os.path.join(all_curriculums_dir_path, exp_no_curr_shaping_folder_name)

serialized_no_curr_sparse_path = os.path.join(exp_no_curr_sparse_path, "serialized_curriculum_lists")
serialized_no_curr_shaping_path = os.path.join(exp_no_curr_shaping_path, "serialized_curriculum_lists")


# exp_with_cl_dir_path = os.path.join(all_models_dir_path, exp_with_cl_dir_name)
# exp_without_cl_dir_path = os.path.join(all_models_dir_path, exp_without_cl_dir_name)
comparison_plots_dir_path = os.path.join(all_curriculums_dir_path, "comparison_plots")
os.makedirs(comparison_plots_dir_path, exist_ok=True)

max_updates_per_env_step = 10


def savePlot(dir_path,
             x_label, no_curr_x_values, curr_x_values,
             y_label, no_curr_y_values, curr_y_values,
             title,
             filename,
             vertical_xs=None,
             y_min=None,
             y_max=None,
             ):
    fig = plt.figure()
    plt.plot(curr_x_values, curr_y_values, 'b', linewidth=0.5, label='with curriculum')
    plt.plot(no_curr_x_values, no_curr_y_values, 'r--', linewidth=0.5, label='no curriculum')
    plt.ylim((y_min, y_max))
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


def savePlots(no_curr_shap_folder_name, no_curr_shap_results_dict, no_curr_shaping_folder_name, no_curr_sparse_results_dict, curriculum_folder_name, curriculum_results_dict):
    no_curr_num = 0
    for no_curriculum_results_dict in [no_curr_shap_results_dict, no_curr_sparse_results_dict]:
        no_curr_num += 1
        if no_curr_num == 1:
            no_curriculum_folder_name = no_curr_shap_folder_name
        elif no_curr_num == 2:
            no_curriculum_folder_name = no_curr_sparse_folder_name

        no_curriculum_undisc_return_per_ep = no_curriculum_results_dict['curriculum_undisc_return_per_ep']
        no_curriculum_num_steps_per_ep = no_curriculum_results_dict['curriculum_num_steps_per_ep']
        no_curriculum_cumul_successes_per_ep = no_curriculum_results_dict['curriculum_cumul_successes_per_ep']
        no_curriculum_epsilon_per_ep = no_curriculum_results_dict['curriculum_epsilon_per_ep']
        no_curriculum_success_step_per_ep = no_curriculum_results_dict['curriculum_success_step_per_ep']
        no_curriculum_test_steps = no_curriculum_results_dict['curriculum_test_steps']
        # no_curriculum_test_episodes = no_curriculum_results_dict['curriculum_test_episodes']
        no_curriculum_test_success_rates = no_curriculum_results_dict['curriculum_test_success_rates']
        no_curriculum_test_mean_returns = no_curriculum_results_dict['curriculum_test_mean_returns']
        no_curriculum_net_updates_per_step = no_curriculum_results_dict['curriculum_net_updates_per_step']
        # no_curriculum_switching_episodes = no_curriculum_results_dict['curriculum_switching_episodes']
        # no_curriculum_switching_steps = no_curriculum_results_dict['curriculum_switching_steps']

        no_curriculum_episodes = range(1, len(no_curriculum_num_steps_per_ep) + 1)
        no_curriculum_steps = range(1, len(no_curriculum_net_updates_per_step) + 1)

        curriculum_undisc_return_per_ep = curriculum_results_dict['curriculum_undisc_return_per_ep']
        curriculum_num_steps_per_ep = curriculum_results_dict['curriculum_num_steps_per_ep']
        curriculum_cumul_successes_per_ep = curriculum_results_dict['curriculum_cumul_successes_per_ep']
        curriculum_epsilon_per_ep = curriculum_results_dict['curriculum_epsilon_per_ep']
        curriculum_success_step_per_ep = curriculum_results_dict['curriculum_success_step_per_ep']
        curriculum_test_steps = curriculum_results_dict['curriculum_test_steps']
        # curriculum_test_episodes = curriculum_results_dict['curriculum_test_episodes']
        curriculum_test_success_rates = curriculum_results_dict['curriculum_test_success_rates']
        curriculum_test_mean_returns = curriculum_results_dict['curriculum_test_mean_returns']
        curriculum_net_updates_per_step = curriculum_results_dict['curriculum_net_updates_per_step']
        curriculum_switching_episodes = curriculum_results_dict['curriculum_switching_episodes']
        curriculum_switching_steps = curriculum_results_dict['curriculum_switching_steps']

        curriculum_episodes = range(1, len(curriculum_num_steps_per_ep) + 1)
        curriculum_steps = range(1, len(curriculum_net_updates_per_step) + 1)

        current_comparison_name = 'comparison_' + no_curriculum_folder_name + '_' + curriculum_folder_name
        current_comparison_dir_path = os.path.join(comparison_plots_dir_path, current_comparison_name)
        os.makedirs(current_comparison_dir_path, exist_ok=True)

        cumul_no_cl_steps = np.cumsum(np.array(no_curriculum_num_steps_per_ep))  # 1 x num episodes
        cumul_cl_steps = np.cumsum(np.array(curriculum_num_steps_per_ep))  # 1 x num episodes

        savePlot(current_comparison_dir_path,
                 'episodes', no_curriculum_episodes, curriculum_episodes,
                 'steps', no_curriculum_num_steps_per_ep, curriculum_num_steps_per_ep,
                 'Environment steps during training',
                 'curriculum_num_steps_per_ep',
                 curriculum_switching_episodes,
                 y_min=-1,
                 y_max=100,
                 )

        # savePlot(current_comparison_dir_path,
        #          'episodes', no_curriculum_episodes, curriculum_episodes,
        #          'returns', no_curriculum_undisc_return_per_ep, curriculum_undisc_return_per_ep,
        #          'Undiscounted returns during training',
        #          'comparison_undisc_return_per_ep',
        #          curriculum_switching_episodes,
        #          )

        savePlot(current_comparison_dir_path,
                 'transitions', cumul_no_cl_steps, cumul_cl_steps,
                 'returns', no_curriculum_undisc_return_per_ep, curriculum_undisc_return_per_ep,
                 'Undiscounted returns during training',
                 'comparison_undisc_return_by_transitions',
                 curriculum_switching_steps,
                 )

        # savePlot(current_comparison_dir_path,
        #          'episodes', no_curriculum_episodes, curriculum_episodes,
        #          'successful episodes', no_curriculum_cumul_successes_per_ep, curriculum_cumul_successes_per_ep,
        #          'Cumulative successful episodes',
        #          'curriculum_cumul_successes_per_ep',
        #          curriculum_switching_episodes,
        #          )

        savePlot(current_comparison_dir_path,
                 'transitions', cumul_no_cl_steps, cumul_cl_steps,
                 'successful episodes', no_curriculum_cumul_successes_per_ep, curriculum_cumul_successes_per_ep,
                 'Cumulative successful episodes during training',
                 'curriculum_cumul_successes_by_transitions',
                 curriculum_switching_steps,
                 )

        # savePlot(current_comparison_dir_path,
        #          'episodes', no_curriculum_episodes, curriculum_episodes,
        #          'epsilon', no_curriculum_epsilon_per_ep, curriculum_epsilon_per_ep,
        #          'Epsilon evolution',
        #          'curriculum_epsilon_per_ep',
        #          curriculum_switching_episodes,
        #          )
        savePlot(current_comparison_dir_path,
                 'transitions', cumul_no_cl_steps, cumul_cl_steps,
                 'epsilon', no_curriculum_epsilon_per_ep, curriculum_epsilon_per_ep,
                 'Epsilon evolution during training',
                 'curriculum_epsilon_by_transitions',
                 curriculum_switching_steps,
                 )

        # savePlot(current_comparison_dir_path,
        #          'episodes', no_curriculum_episodes, curriculum_episodes,
        #          'steps', no_curriculum_success_step_per_ep, curriculum_success_step_per_ep,
        #          'Steps to reach a goal state',
        #          'curriculum_success_step_per_ep',
        #          curriculum_switching_episodes,
        #          )

        savePlot(current_comparison_dir_path,
                 'transitions', cumul_no_cl_steps, cumul_cl_steps,
                 'steps to goal state', no_curriculum_success_step_per_ep, curriculum_success_step_per_ep,
                 'Steps to reach a goal state during training',
                 'curriculum_success_step_by_transitions',
                 curriculum_switching_steps,
                 )

        savePlot(current_comparison_dir_path,
                 'transitions', no_curriculum_test_steps, curriculum_test_steps,
                 'success rate', no_curriculum_test_success_rates, curriculum_test_success_rates,
                 'Success rate in test conditions',
                 'curriculum_success_rate_by_transitions',
                 curriculum_switching_steps,
                 y_min=-0.1,
                 y_max=1.1,
                 )

        # savePlot(current_comparison_dir_path,
        #          'episodes', no_curriculum_test_episodes, curriculum_test_episodes,
        #          'success rate', no_curriculum_test_success_rates, curriculum_test_success_rates,
        #          'Success rate in test conditions',
        #          'curriculum_success_rate_per_ep',
        #          curriculum_switching_episodes,
        #          )

        savePlot(current_comparison_dir_path,
                 'transitions', no_curriculum_test_steps, curriculum_test_steps,
                 'mean return', no_curriculum_test_mean_returns, curriculum_test_mean_returns,
                 'Mean undisc. return in test conditions',
                 'curriculum_test_mean_returns_by_transitions',
                 curriculum_switching_steps,
                 )

        # savePlot(current_comparison_dir_path,
        #          'transitions', no_curriculum_test_steps, curriculum_test_steps,
        #          'mean return', no_curriculum_test_mean_returns, curriculum_test_mean_returns,
        #          'Mean undisc. return in test conditions',
        #          'curriculum_test_mean_returns_by_transitions',
        #          curriculum_switching_steps,
        #          )

        savePlot(current_comparison_dir_path,
                 'transitions', no_curriculum_steps, curriculum_steps,
                 'updates', no_curriculum_net_updates_per_step, curriculum_net_updates_per_step,
                 'Number of network updates per env. step',
                 'curriculum_net_updates_by_transitions',
                 curriculum_switching_steps,
                 y_min=-0,
                 y_max=max_updates_per_env_step + 1,
                 )


def get_no_curr_prev_runs(serialized_lists_path):
    results_dict = {}
    lists_to_deserialize = ['curriculum_cumul_successes_per_ep',
                            'curriculum_epsilon_per_ep',
                            'curriculum_net_updates_per_step',
                            'curriculum_num_steps_per_ep',
                            'curriculum_success_step_per_ep',
                            'curriculum_switching_episodes',
                            'curriculum_switching_steps',
                            'curriculum_test_episodes',
                            'curriculum_test_mean_returns',
                            'curriculum_test_steps',
                            'curriculum_test_success_rates',
                            'curriculum_undisc_return_per_ep',
                            ]  # have to be lists
    for list_to_deserialize in lists_to_deserialize:
        list_json_file = os.path.join(serialized_lists_path, list_to_deserialize + '.json')
        with open(list_json_file, "r") as json_file:
            results_dict[list_json_file] = json.load(json_file)

    return results_dict

# ###################
no_curriculum_shaping = Curriculum.NO_CURRICULUM_SHAPING
no_curriculum_sparse = Curriculum.NO_CURRICULUM_SPARSE

Curriculums = [Curriculum.CURRICULUM_INITIALIZE_FURTHER_SPARSE,
               Curriculum.CURRICULUM_INITIALIZE_FURTHER_SHAPING,
               Curriculum.CURRICULUM_DECREASING_SPEED_SPARSE,
               Curriculum.CURRICULUM_DECREASING_SPEED_SHAPING,
               Curriculum.CURRICULUM_INCREASING_JOINT_NUMBER_SPARSE,
               Curriculum.CURRICULUM_INCREASING_JOINT_NUMBER_SHAPING,
               ]

task = RobotEnv.TASK_REACH_CUBE

testing_scripts = True  # ##################
load_no_curriculum_prev_results = True  # ################
# max_steps_per_episode = 200
num_episodes = 5000  # aproximate
max_steps_per_ep = 50  # aproximate
max_total_transitions = num_episodes * max_steps_per_ep  # episodes x max_steps_per_ep
num_hidden_layers = 3
num_neurons_per_hidden = 50

batch_size = 32
lrate = 1e-4
# replay_start_size = (num_episodes // 20) * max_steps_per_episode
# replay_memory_size = 10 * replay_start_size
disable_saving = True
sync_mode = True
portNb = 19999

# ################

curr_args = dict(task=task,
                 # max_steps_per_episode=max_steps_per_episode,
                 # num_episodes=num_episodes,
                 max_total_transitions=max_total_transitions,
                 num_hidden_layers=num_hidden_layers,
                 num_neurons_per_hidden=num_neurons_per_hidden,
                 batch_size=batch_size,
                 lrate=lrate,
                 testing_scripts=testing_scripts,  # ##
                 max_updates_per_env_step=max_updates_per_env_step,
                 # replay_start_size=replay_start_size,
                 # replay_memory_size=replay_memory_size,
                 disable_saving=disable_saving,
                 sync_mode=sync_mode,
                 portNb=portNb,
                 )

if load_no_curriculum_prev_results:
    no_curr_sparse_folder_name = exp_no_curr_sparse_folder_name
    no_curr_shap_folder_name = exp_no_curr_shaping_folder_name
    no_curr_shap_results_dict = get_no_curr_prev_runs(serialized_no_curr_shaping_path)
    no_curr_sparse_results_dict = get_no_curr_prev_runs(serialized_no_curr_sparse_path)

else:
    curr_args.update(dict(curriculum=no_curriculum_shaping))
    no_curr_shap = Curriculum(**curr_args)
    no_curr_shap_results_dict = no_curr_shap.run()
    no_curr_shap_folder_name = no_curr_shap.folder_name

    curr_args.update(dict(curriculum=no_curriculum_sparse))
    no_curr_sparse = Curriculum(**curr_args)
    no_curr_sparse_results_dict = no_curr_sparse.run()
    no_curr_sparse_folder_name = no_curr_sparse.folder_name

for curriculum in Curriculums:
    curr_args.update(dict(curriculum=curriculum))
    curr = Curriculum(**curr_args)
    curriculum_results_dict = curr.run()
    curriculum_folder_name = curr.folder_name
    savePlots(no_curr_shap_folder_name, no_curr_shap_results_dict,
              no_curr_sparse_folder_name, no_curr_sparse_results_dict,
              curriculum_folder_name, curriculum_results_dict)
