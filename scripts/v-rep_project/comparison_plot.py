# import os
# import json
# import numpy as np
# from matplotlib import pyplot as plt

# exp_with_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE
# exp_without_cl_dir_name = 'cl_decreasing_speeds_2017-Aug-10_16-01-01'  # #######TO CHANGE

# current_dir_path = os.path.dirname(os.path.realpath(__file__))
# all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
# exp_with_cl_dir_path = os.path.join(all_models_dir_path, exp_with_cl_dir_name)
# exp_without_cl_dir_path = os.path.join(all_models_dir_path, exp_without_cl_dir_name)

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

    def savePlot(self,
                 dir_path,
                 x1_label, x1_values,
                 y1_label, y1_values,
                 x2_label, x2_values,
                 y2_label, y2_values,
                 title,
                 filename,
                 vertical_xs=None
                 ):
        fig = plt.figure()
        # start at episode 1
        plt.plot(x_values, y_values, linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if vertical_xs is not None:
            # vertical lines at subtask switching
            for vertical_x in vertical_xs:
                plt.axvline(x=vertical_x, ls='dashed', color='r')
        plot_file = os.path.join(dir_path, filename + '.svg')
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close()

from evaluate_curriculum import Curriculum

no_curriculum = Curriculum.NO_CURRICULUM_VEL_1

curr = Curriculum(curriculum=no_curriculum,
                  task=Curriculum.TASK_REACH_CUBE,
                  max_steps_per_episode=200,
                  num_episodes=2000,
                  num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  batch_size=32,
                  lrate=1e-4,
                  testing_scripts=True,  # ##
                  max_updates_per_env_step=10,
                  )
no_curriculum_results_dict = curr.run()

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
no_curriculum_switching_eps = no_curriculum_results_dict['curriculum_switching_eps']
no_curriculum_switching_steps = no_curriculum_results_dict['curriculum_switching_steps']

for curriculum in Curriculums:
    curr = Curriculum(curriculum=curriculum,
                      task=Curriculum.TASK_REACH_CUBE,
                      max_steps_per_episode=200,
                      num_episodes=2000,
                      num_hidden_layers=2,
                      num_neurons_per_hidden=50,
                      batch_size=32,
                      lrate=1e-4,
                      testing_scripts=True,  # ##
                      max_updates_per_env_step=10,
                      )
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
    curriculum_switching_eps = results_dict['curriculum_switching_eps']
    curriculum_switching_steps = results_dict['curriculum_switching_steps']