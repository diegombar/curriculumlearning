import training_independent_joints as training
import time
import os
import numpy as np
from matplotlib import pyplot as plt


def plot_cl_success_rate(dir_path, subtask_switch_episodes, test_success_rate, test_step_numbers):
    fig = plt.figure()
    plt.plot(test_step_numbers, test_success_rate, linewidth=0.5)
    plt.ylabel('steps')
    plt.xlabel('success rate')
    plt.title('Success rate in test conditions')
    #vertical lines at subtask switching
    for switch_ep in subtask_switch_episodes:
        plt.axvline(x=switch_ep)
    plot_file = os.path.join(dir_path, 'success_rate.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()


def plot_cl_cumul_successes(dir_path, subtask_switch_episodes, cumul_successes):
    fig = plt.figure()
    episodes = range(1, len(cumul_successes) + 1)  # start at episode 1
    plt.plot(episodes, cumul_successes, linewidth=0.5)
    plt.ylabel('episodes')
    plt.xlabel('cumulative successes')
    plt.title('Cumulative successes during training')
    #vertical lines at subtask switching
    for switch_ep in subtask_switch_episodes:
        plt.axvline(x=switch_ep)
    plot_file = os.path.join(dir_path, 'cumul_successes.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()




timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")

# tasks

TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

# curriculums

NO_CURRICULUM_VEL_025 = 0
NO_CURRICULUM_VEL_1 = 3
CURRICULUM_DECREASING_SPEED = 1
CURRICULUM_INCREASING_JOINT_NUMBER = 2
#CURRICULUM_STARTING_AWAY_FROM_TARGET = 3

################# CHOOSE ################

curriculum = CURRICULUM_INCREASING_JOINT_NUMBER ##############
task = TASK_REACH_CUBE #########
testing_scripts = True  # set to True test scripts for a few episodes/steps

#########################################

episodes = 5000
Velocities = [0.25]
NumOfJoints = [6]
if curriculum == NO_CURRICULUM_VEL_025:
    experiment_name = "no_curriculum_vel_025"
    success_rate_for_subtask_completion = False
elif curriculum == NO_CURRICULUM_VEL_1:
    experiment_name = "no_curriculum_vel_1"
    Velocities = [1.0]
    success_rate_for_subtask_completion = False
elif curriculum == CURRICULUM_DECREASING_SPEED:
    experiment_name = "cl_decreasing_speeds"
    Velocities = [1, 0.5, 0.25]
    episodes = episodes // len(Velocities)
    success_rate_for_subtask_completion = True
elif curriculum == CURRICULUM_INCREASING_JOINT_NUMBER:
    experiment_name = "cl_increasing_num_of_joints"
    NumOfJoints = range(1, 7)
    episodes = episodes // len(NumOfJoints)
    success_rate_for_subtask_completion = True

if testing_scripts:
    experiment_name = "TEST"

max_steps = 200

folder_name =  experiment_name + '_' + timestr
experiment_dir_path = os.path.join(all_models_dir_path, folder_name)

# vel025 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")

targetRelativePos = (0.0, 0.5) #relative x, y in metres

subt_initial_step = 0
cl_switching_eps = []
cl_cumul_successes = np.array([])
cl_test_success_rates = [0.0]
cl_test_steps = [0]
cl_total_time = 0.0

model_path = None
st_num = 0

trainDQL_args = dict(
                    experiment_dir_path=experiment_dir_path,
                    num_hidden_layers=2,
                    num_neurons_per_hidden=50,
                    num_episodes=episodes,  #400
                    max_steps_per_episode=max_steps,  #200
                    e_min=0.01,
                    task=task,
                    model_saving_period=episodes // 5,
                    lrate=1E-3,  # 1E-3 seems to work fine
                    batch_size=32,
                    replay_start_size=50000,
                    replay_memory_size=500000,
                    showGUI=True,
                    velocity=0,  # 1.0 seems to work fine
                    model_to_load_file_path=model_path,
                    use_variable_names=True,
                    skip_training=False,
                    notes=experiment_name,
                    previous_norm=False,
                    targetRelativePos=targetRelativePos,
                    policy_test_period=100,  # episodes
                    success_rate_for_subtask_completion=success_rate_for_subtask_completion,  # change with/without CL
                    nSJoints=6,
                    nAJoints=6
                    )

if testing_scripts:
    trainDQL_args.update(dict(
                              num_episodes=10,
                              max_steps_per_episode=2,
                              model_saving_period=2,
                              batch_size=1,
                              replay_start_size=6,
                              replay_memory_size=10,
                              policy_test_period=5,
                             )
                        )

for vel in Velocities:
    for nASJoints in NumOfJoints:
        st_num += 1
        trainDQL_args.update(dict(
                              velocity=vel,
                              nSJoints=nASJoints,
                              nAJoints=nASJoints,
                              model_to_load_file_path=model_path,
                             )
                        )

        model_path, subt_total_steps, subt_cumul_successes, subt_test_success_rates, subt_test_steps, st_time = training.trainDQL(**trainDQL_args)

        # update switching steps
        last_abs_step = subt_total_steps + subt_initial_step
        cl_switching_eps.append(last_abs_step)

        # update cumulative successes
        abs_subt_cumul_successes = np.array(subt_cumul_successes) + cl_cumul_successes[-1]  # subtask counter to cl counter
        cl_cumul_successes = np.concatenate((cl_cumul_successes, abs_subt_cumul_successes), axis=0)

        # update test success rates
        cl_test_success_rates += subt_test_success_rates  # list concatenation

        # update test steps
        abs_subt_test_steps = np.array(subt_test_steps) + last_abs_step  # subtask step to cl step
        cl_test_steps = np.concatenate((cl_test_steps, abs_subt_test_steps), axis=0)

        # update curriculum time
        cl_total_time += st_time # in hours

        os.makedirs(experiment_dir_path, exist_ok=True)
        plot_cl_cumul_successes(experiment_dir_path, cl_switching_eps, cl_cumul_successes)
        plot_cl_success_rate(experiment_dir_path, cl_switching_eps, cl_test_success_rates, cl_test_steps)

        subt_initial_step = last_abs_step

        stats_dict = {"number_of_steps_executed_curriculum_so_far":last_abs_step}
        stats_dict["number_of_episodes_executed_curriculum_so_far"] = len(cl_cumul_successes)
        stats_dict["training_time_in_hours_curriculum_so_far"] = cl_total_time
        stats_file_path = os.path.join(experiment_dir_path, "stats_sub_task_" + str(st_num) + ".txt")
        with open(stats_file_path, "w") as stats_file:
            json.dump(stats_dict, stats_file, sort_keys=True, indent=4)

end_stats_dict = {"total_number_of_steps_executed_curriculum":last_abs_step}
end_stats_dict["total_number_of_episodes_executed_curriculum"] = len(cl_cumul_successes)
end_stats_dict["total_training_time_in_hours_curriculum"] = cl_total_time
stats_file_path = os.path.join(experiment_dir_path, "end_stats.txt")
with open(stats_file_path, "w") as stats_file:
    json.dump(end_stats_dict, stats_file, sort_keys=True, indent=4)
