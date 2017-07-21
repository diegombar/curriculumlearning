import training_independent_joints as training
import os.path

#feed set of hyper params

for i in range(1,8):
      n = i*400
      print('\ntesting num_episodes=', i*2)
      training.trainDQL(num_hidden_layers=2,
                        num_neurons_per_hidden=50,
                        num_episodes=n, #3000
                        max_steps_per_episode=300,
                        e_min=0.1,
                        model_saving_period=200,
                        batch_size=32,
                        replay_start_size=50000,
                        replay_memory_size=500000,
                        showGUI=True,
                        velocity=0.3,
                        model_to_load_file_path=None,
                        notes="testing different numbers of episodes, with 300 steps per ep",
                        previous_norm=False)
