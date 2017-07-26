import training_independent_joints as training
import os.path

##feed set of hyper params
for i in range(1, 4):
    n = i * 400
    for j in range(0,10):
        vel = (10 - j)
        trained_model_path = training.trainDQL(num_hidden_layers=2,
                          num_neurons_per_hidden=50,
                          num_episodes=n,#3000
                          max_steps_per_episode=300,
                          e_min=0.1,
                          model_saving_period=400,
                          lrate=1E-3, # 1E-3 seems to work fine
                          batch_size=32,
                          replay_start_size=50000,
                          replay_memory_size=500000,
                          showGUI=True,
                          velocity=vel, # 1.0 seems to work fine
                          model_to_load_file_path=None,
                          notes="testing different numbers of episodes, with 300 steps per ep, lrate=1E-3, different speeds",
                          previous_norm=False)
