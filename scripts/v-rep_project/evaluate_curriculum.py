import training_independent_joints as training


## use the following to test the training script
# training.trainDQL(num_hidden_layers=2,
#                   num_neurons_per_hidden=50,
#                   num_episodes=100,
#                   max_steps_per_episode=2,
#                   e_min=0.01,
#                   model_saving_period=2,
#                   batch_size=2,
#                   replay_start_size=2,
#                   replay_memory_size=10,
#                   showGUI=False,
#                   velocity=0.3,
#                   model_to_load_file_path=None,
#                   notes="testing algorithm")

#feed set of hyper params
training.trainDQL(num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  num_episodes=2000,
                  max_steps_per_episode=500,
                  e_min=0.01,
                  model_saving_period=100,
                  batch_size=32,
                  replay_start_size=50000,
                  replay_memory_size=500000,
                  showGUI=True,
                  velocity=0.3,
                  model_to_load_file_path=None,
                  notes="changed rewards")
