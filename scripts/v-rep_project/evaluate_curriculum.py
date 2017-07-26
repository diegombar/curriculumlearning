import training_independent_joints as training

##feed set of hyper params
episodes_per_vel = 400 
max_steps= 300
saved_model_path = training.trainDQL(num_hidden_layers=2,
                      num_neurons_per_hidden=50,
                      num_episodes=episodes_per_vel,
                      max_steps_per_episode=max_steps,
                      e_min=0.01,
                      model_saving_period=episodes_per_vel,
                      lrate=1E-3, # 1E-3 seems to work fine
                      batch_size=32,
                      replay_start_size=50000,
                      replay_memory_size=500000,
                      showGUI=True,
                      velocity=vel, # 1.0 seems to work fine
                      model_to_load_file_path=None,
                      use_variable_names=True,
                      notes="curriculum learning decreasing speed",
                      previous_norm=False)

for vel in [2, 1, 0.5, 0.25]:
    previous_saved_model_path = saved_model_path
    saved_model_path = training.trainDQL(num_hidden_layers=2,
                          num_neurons_per_hidden=50,
                          num_episodes=episodes_per_vel,#400
                          max_steps_per_episode=max_steps,
                          e_min=0.01,
                          model_saving_period=episodes_per_vel,
                          lrate=1E-3, # 1E-3 seems to work fine
                          batch_size=32,
                          replay_start_size=50000,
                          replay_memory_size=500000,
                          showGUI=True,
                          velocity=vel, # 1.0 seems to work fine
                          model_to_load_file_path=previous_saved_model_path,
                          use_variable_names=True,
                          skip_training=False,
                          notes="curriculum learning decreasing speed",
                          previous_norm=False)