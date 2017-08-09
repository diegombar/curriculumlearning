import training_independent_joints as training
import os
import time
from matplotlib import pyplot as plt

timestr = time.strftime("%Y-%b-%d_%H-%M-%S", time.gmtime())
current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
experiment_name = "testing_algorithm"
folder_name = experiment_name + '_' + timestr
experiment_dir_path = os.path.join(all_models_dir_path, folder_name)

# tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2


def plot_success(dir_path, test_success_rate, test_step_numbers):
    fig = plt.figure()
    plt.plot(test_step_numbers, test_success_rate, linewidth=0.5)
    plt.ylabel('steps')
    plt.xlabel('success rate')
    plt.title('Success rate in test conditions')
    plot_file = os.path.join(dir_path, 'with_curriculum_success_rate.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()


# nASJoints = 6
targetRelativePos = (0.0, 0.7)  # relative x, y in metres

saved_model_path = None
saved_test_success_rate_list = None
saved_test_step_numbers = None

# test the training script
for nASJoints in range(1, 7):
    saved_model_path, saved_test_success_rate_list, saved_test_step_numbers = training.trainDQL(
                              experiment_folder_name=folder_name,
                              num_hidden_layers=2,
                              num_neurons_per_hidden=50,
                              num_episodes=10,#400
                              max_steps_per_episode=2,
                              e_min=0.1,
                              task=TASK_REACH_CUBE,
                              model_saving_period=2,
                              lrate=1E-3,  # 1E-3 seems to work fine
                              batch_size=1,
                              replay_start_size=6,
                              replay_memory_size=10,
                              showGUI=True,
                              velocity=1.0,  # 1.0 seems to work fine
                              model_to_load_file_path=saved_model_path,
                              use_variable_names=True,
                              skip_training=False,
                              notes=experiment_name,
                              previous_norm=False,
                              #targetRelativePos=targetRelativePos,
                              policy_test_period=5,
                              test_success_rate_list=saved_test_success_rate_list,
                              test_step_numbers=saved_test_step_numbers,
                              success_rate_for_subtask_completion=False,  # change with/without CL
                              nSJoints=nASJoints,
                              nAJoints=nASJoints
                              )
    os.makedirs(experiment_dir_path, exist_ok=True)
    plot_success(experiment_dir_path, saved_test_success_rate_list, saved_test_step_numbers)
