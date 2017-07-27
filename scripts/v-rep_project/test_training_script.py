import training_independent_joints as training
import os.path
import time


timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()

#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

targetRelativePos = (0.0, 0.7) #relative x, y in metres

# test the training script
training.trainDQL(experiment_folder_name='testing_algorithm_'+timestr,
                  num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  num_episodes=10,
                  max_steps_per_episode=2,
                  e_min=0.1,
                  task=TASK_PUSH_CUBE_TO_TARGET_POSITION,
                  model_saving_period=2,
                  batch_size=2,
                  replay_start_size=2,
                  replay_memory_size=10,
                  showGUI=True,
                  velocity=1.0,
                  model_to_load_file_path=None,
                  use_variable_names=True,
                  notes="testing algorithm",
                  targetRelativePos=targetRelativePos)
