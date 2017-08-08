import training_independent_joints as training
import time
import os
from matplotlib import pyplot as plt


timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")


#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

##feed set of hyper params

def plot_success(dir_path, test_success_rate, test_step_numbers):
    fig = plt.figure()
    plt.plot(test_step_numbers, test_success_rate, linewidth=0.5)
    plt.ylabel('steps')
    plt.xlabel('success rate')
    plt.title('Success rate in test conditions')
    plot_file = os.path.join(dir_path, 'with_curriculum_success_rate.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()

################# EDIT EXPERIMENT NAME #############

experiment_name = "CL_success_rate_to_decrease_vel_TEST"

####################################################

folder_name =  experiment_name + '_' + timestr
<<<<<<< HEAD

episodes = 4000
=======
episodes = 5000
>>>>>>> 5e10992cacf42d71be3699d171301b9f79ee4595
max_steps= 200

#targetRelativePos = (0.0, 0.5) #relative x, y in metres

experiment_dir_path = os.path.join(all_models_dir_path, folder_name)

# vel025 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")

saved_model_path = None
saved_test_success_rate_list = None
saved_test_step_numbers = None

#for vel in [1, 0.5, 0.25]:
for vel in [0.25]:
    saved_model_path, saved_test_success_rate_list, saved_test_step_numbers = training.trainDQL(
                              experiment_folder_name=folder_name,
                              num_hidden_layers=2,
                              num_neurons_per_hidden=50,
                              num_episodes=episodes,#400
                              max_steps_per_episode=max_steps,
                              e_min=0.01,
                              task=TASK_REACH_CUBE,
                              model_saving_period=episodes//5,
                              lrate=1E-3, # 1E-3 seems to work fine
                              batch_size=32,
                              replay_start_size=50000,
                              replay_memory_size=500000,
                              showGUI=True,
                              velocity=vel, # 1.0 seems to work fine
                              model_to_load_file_path=saved_model_path,
                              use_variable_names=True,
                              skip_training=False,
                              notes=experiment_name,
                              previous_norm=False,
                              #targetRelativePos=targetRelativePos,
                              policy_test_period = 100,
                              test_success_rate_list=saved_test_success_rate_list,
                              test_step_numbers=saved_test_step_numbers,
                              success_rate_for_subtask_completion=False # change with/without CL
                              )
    os.makedirs(experiment_dir_path, exist_ok=True)
    plot_success(experiment_dir_path, saved_test_success_rate_list, saved_test_step_numbers)