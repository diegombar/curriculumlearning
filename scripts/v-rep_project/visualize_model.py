import os.path
import time
from robotenv import RobotEnv
from dql_algorithm import DQLAlgorithm

timestr = time.strftime("%b-%d_%H-%M-%S", time.gmtime())  # or time.localtime()

current_dir_path = os.path.dirname(os.path.realpath(__file__))
experiments_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
experiment_name = 'visualization'
folder_name = timestr + '_' + experiment_name
vis_experiment_dir_path = os.path.join(experiments_dir_path, folder_name)


# model to load

# model1_ep400 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","saved_checkpoints","checkpoint_model-400")

# model1_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","trained_model","final_model-3000") #success

# model2_ep400 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","saved_checkpoints","checkpoint_model-400")


# model2_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-11_14-19-11","trained_model","final_model-3000")

# model3_ep900 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-12_01-32-42","saved_checkpoints","checkpoint_model-900")


# model3_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-12_01-32-42","trained_model","final_model-3000")

# new_model = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-26_09-41-08","trained_model","final_model-400")

# vel2 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_17-00-30_vel=2","trained_model","final_model-400")
# vel1= os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_20-30-32_vel=1","trained_model","final_model-400")
# vel05 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_23-38-21_vel=05","trained_model","final_model-400")
# vel025 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")

# model_1600ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-22_21-57-58_1600ep/trained_model",
#    "final_model-1600") #good results with e_min=0.1

# model_1600ep_vel1_1 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-25_02-48-01_1600ep/trained_model",
#    "final_model-1600")

# model_2800ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-24_07-19-16_2800ep/trained_model",
#    "final_model-2800")

# model_2000ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-25_12-02-10_2000ep/trained_model",
#    "final_model-2000") #good results with e_min=0.1


# model_800ep_vel1_emin0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "e_min=0, different num_ep, steps=200, vel=1_2017-Jul-26_22-22-21/model_and_results_2017-Jul-27_02-26-31/trained_model",
#    "final_model-800")

# reach_then_push = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "CL_reach_then_push_2017-Jul-26_20-27-48/model_and_results_2017-Jul-27_07-30-58",
#    "trained_model",
#    "final_model-400")

# model_longer_training_025 = os.path.join(experiments_dir_path,
#    "longer_training_vel025_2017-Jul-27_17-29-43/model_and_results_2017-Jul-27_17-29-43/trained_model",
#    "final_model-1000")

# cl_decreasing_speeds = os.path.join(experiments_dir_path,
#                                     "cl_decreasing_speeds_2017-Aug-10_21-22-07/model_and_results_2017-Aug-10_21-22-07/trained_model",
#                                     "final_model-1666")

# no_cl_5000_vel025_new = os.path.join(experiments_dir_path,
#                                      "no_curriculum_vel_025_2017-Aug-12_19-01-58/model_and_results_2017-Aug-12_19-01-58/trained_model",
#                                      "final_model-5000")  # 3 hlayers x 100 neurons

# no_cl_5000_vel025_old = os.path.join(experiments_dir_path,
#                                      "no_curriculum_vel_025_2017-Aug-11_16-24-22/model_and_results_2017-Aug-11_16-24-22/trained_model",
#                                      "final_model-5000")  # 2 hlayers x 50 neurons

# no_cl_5000_vel1 = os.path.join(experiments_dir_path,
#                                "no_curriculum_vel_025_2017-Aug-11_16-24-22/model_and_results_2017-Aug-11_16-24-22/trained_model",
#                                "final_model-5000")  # 2 hlayers x 50 neurons
# pushing_1 = os.path.join(experiments_dir_path,
#                          "2017-Aug-26_06-04-58_pushing_cl_increasing_num_of_joints/model_and_results_2017-Aug-26_12-23-00/trained_model",
#                          "final_model-166")  # 2 hlayers x 50 neurons


# tasks

# TASK_REACH_CUBE = 1
# TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

# ##### CHOOSE


reaching_no_cl_shaping = os.path.join(experiments_dir_path,
                                      "Sep-01_14-53-22_graphic04.doc.ic.ac.uk_reaching_no_cl_shaping/model_and_results_Sep-01_14-53-22/trained_model",
                                      "final_model-5000")  # 2 hlayers x 50 neurons

reaching_no_cl_sparse = os.path.join(experiments_dir_path,
                                     "Sep-02_02-39-59_graphic04.doc.ic.ac.uk_reaching_no_cl_sparse/model_and_results_Sep-02_02-39-59/trained_model",
                                     "final_model-5000")  # 2 hlayers x 50 neurons

reaching_no_cl_shaping_point03 = os.path.join(experiments_dir_path,
                                              "point03/Sep-04_16-35-41_point03.doc.ic.ac.uk_reaching_no_cl_shaping/model_and_results_Sep-04_16-35-41/trained_model",
                                              "final_model-555")  # 2 hlayers x 50 neurons

reaching_no_cl_sparse_point03 = os.path.join(experiments_dir_path,
                                             "point03/Sep-04_18-46-32_point03.doc.ic.ac.uk_reaching_no_cl_sparse/model_and_results_Sep-04_18-46-32/trained_model",
                                             "final_model-555")  # 2 hlayers x 50 neurons

pushing_no_cl_shaping_matrix18 = os.path.join(experiments_dir_path,
                                              "matrix18/Sep-04_17-06-32_matrix18.doc.ic.ac.uk_pushing_no_cl_shaping/model_and_results_Sep-04_17-06-32/trained_model",
                                              "final_model-454")  # 2 hlayers x 50 neurons

pushing_no_cl_sparse_matrix18 = os.path.join(experiments_dir_path,
                                             "matrix18/Sep-04_19-51-35_matrix18.doc.ic.ac.uk_pushing_no_cl_sparse/model_and_results_Sep-04_19-51-35/trained_model",
                                             "final_model-454")  # 2 hlayers x 50 neurons

model_to_load = pushing_no_cl_shaping_matrix18


task = RobotEnv.TASK_PUSH_CUBE_TO_TARGET_POSITION
vel = 1.0
targetCubePosition = (0.15, 0.35)
max_steps_per_episode = 80
num_episodes = 10
max_total_transitions = num_episodes * max_steps_per_episode
model_saving_period = num_episodes * 10
num_hidden_layers = 3
num_neurons_per_hidden = 50
max_updates_per_env_step = 10
batch_size = 32
lrate = 1e-4
replay_start_size = (num_episodes // 20) * max_steps_per_episode
replay_memory_size = 10 * replay_start_size
disable_saving = True
sync_mode = True
portNb = 19999
test_max_steps_per_episode = 90

trainDQL_args = dict(experiment_dir_path=vis_experiment_dir_path,
                     num_hidden_layers=num_hidden_layers,
                     num_neurons_per_hidden=num_neurons_per_hidden,
                     # num_episodes=num_episodes,  # 400
                     max_total_transitions=max_total_transitions,
                     max_steps_per_episode=max_steps_per_episode,  # 200
                     e_min=0.01,
                     task=task,
                     # model_saving_period=model_saving_period,
                     # lrate=1E-5,  # 1E-3 seems to work fine
                     # batch_size=32,
                     # replay_start_size=50000,
                     # replay_memory_size=500000,
                     showGUI=True,
                     velocity=vel,  # 1.0 seems to work fine
                     model_to_load_file_path=model_to_load,
                     use_variable_names=True,
                     skip_training=True,
                     notes=experiment_name,
                     previous_norm=False,  # note: use previous_norm for first few models (angles were normalized to [0,2] (now fixed))
                     targetCubePosition=targetCubePosition,
                     # nSJoints=6,
                     nAJoints=6,
                     portNb=portNb,
                     old_bias=False,  # note: use old_bias for some past models (scalar biases) (now fixed))
                     disable_saving=disable_saving,
                     sync_mode=sync_mode,
                     test_max_steps_per_episode=test_max_steps_per_episode,
                     # policy_test_period=100,  # episodes
                     # policy_test_episodes=20,  # episodes
                     # max_updates_per_env_step=self.max_updates_per_env_step,
                     )

# curr_args = dict(curriculum=curriculum,
#                  task=task,
#                  # max_steps_per_episode=max_steps_per_episode,
#                  # num_episodes=num_episodes,
#                  max_total_transitions=max_total_transitions,
#                  num_hidden_layers=num_hidden_layers,
#                  num_neurons_per_hidden=num_neurons_per_hidden,
#                  batch_size=batch_size,
#                  lrate=lrate,
#                  testing_scripts=testing_scripts,  # ##
#                  max_updates_per_env_step=max_updates_per_env_step,
#                  # replay_start_size=replay_start_size,
#                  # replay_memory_size=replay_memory_size,
#                  disable_saving=disable_saving,
#                  sync_mode=sync_mode,
#                  portNb=portNb,
#                  )

dql = DQLAlgorithm(**trainDQL_args)

dql.run()
