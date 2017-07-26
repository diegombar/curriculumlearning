import training_independent_joints as training
import time

timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

##feed set of hyper params
experiment_name = 'e_min=0, different num_ep, steps=200, vel=1'
folder_name =  experiment_name + '_' + timestr
# episodes = 400
max_steps= 200
targetRelativePos = (0.0, 0.5) #relative x, y in metres

for episodes in [200, 400, 800, 1200, 1600]:
      saved_model_path = training.trainDQL(
                              experiment_folder_name=folder_name,
                              num_hidden_layers=2,
                              num_neurons_per_hidden=50,
                              num_episodes=episodes,
                              max_steps_per_episode=max_steps,
                              e_min=0.0,
                              task=TASK_REACH_CUBE,
                              model_saving_period=episodes//2,
                              lrate=1E-3, # 1E-3 seems to work fine
                              batch_size=32,
                              replay_start_size=50000,
                              replay_memory_size=500000,
                              showGUI=True,
                              velocity=1.0, # 1.0 seems to work fine
                              model_to_load_file_path=None,
                              use_variable_names=True,
                              notes="e_min=0, different num_ep, steps=200, vel=1",
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
#                               task=TASK_PUSH_CUBE_TO_TARGET_POSITION,
#                               model_saving_period=episodes//2,
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
