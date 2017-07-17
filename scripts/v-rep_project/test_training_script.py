import training_independent_joints as training
import os.path


# test the training script
training.trainDQL(num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  num_episodes=10,
                  max_steps_per_episode=2,
                  e_min=0.1,
                  model_saving_period=2,
                  batch_size=2,
                  replay_start_size=2,
                  replay_memory_size=10,
                  showGUI=True,
                  velocity=0.3,
                  model_to_load_file_path=None,
                  use_variable_names=True,
                  notes="testing algorithm")
