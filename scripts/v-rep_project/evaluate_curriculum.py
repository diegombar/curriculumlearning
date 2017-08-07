import training_independent_joints as training
import time
import os

timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_dir_path = os.path.dirname(os.path.realpath(__file__))

#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

##feed set of hyper params
experiment_name = 'longer_training_vel025'
folder_name =  experiment_name + '_' + timestr
episodes = 1000
max_steps= 200
targetRelativePos = (0.0, 0.5) #relative x, y in metres

vel025 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")



saved_model_path = training.trainDQL(
                        experiment_folder_name=folder_name,
                        num_hidden_layers=2,
                        num_neurons_per_hidden=50,
                        num_episodes=episodes,
                        max_steps_per_episode=max_steps,
                        e_min=0.01,
                        task=TASK_REACH_CUBE,
                        model_saving_period=episodes//5,
                        lrate=1E-5, # 1E-3 seems to work fine
                        batch_size=32,
                        replay_start_size=50000,
                        replay_memory_size=500000,
                        showGUI=True,
                        velocity=0.25, # 1.0 seems to work fine
                        model_to_load_file_path=vel025,
                        use_variable_names=True,
                        skip_training=False,
                        notes="longer_training_vel025",
                        previous_norm=False)

# for vel in [1, 0.5, 0.25]:
#     previous_saved_model_path = saved_model_path
#     saved_model_path = training.trainDQL(
#                               experiment_folder_name=folder_name,
#                               num_hidden_layers=2,
#                               num_neurons_per_hidden=50,
#                               num_episodes=episodes,#400
#                               max_steps_per_episode=max_steps,
#                               e_min=0.01,
#                               task=TASK_REACH_CUBE,
#                               model_saving_period=episodes//4,
#                               lrate=1E-3, # 1E-3 seems to work fine
#                               batch_size=32,
#                               replay_start_size=50000,
#                               replay_memory_size=500000,
#                               showGUI=True,
#                               velocity=vel, # 1.0 seems to work fine
#                               model_to_load_file_path=previous_saved_model_path,
#                               use_variable_names=True,
#                               skip_training=False,
#                               notes="curriculum learning tasks reach then push",
#                               previous_norm=False,
#                               targetRelativePos=targetRelativePos)
