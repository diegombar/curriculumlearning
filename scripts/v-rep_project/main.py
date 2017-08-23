from evaluate_curriculum import Curriculum
from robotenv import RobotEnv

testing_scripts = False
max_steps_per_episode = 200
num_episodes = 2000
num_hidden_layers = 2
num_neurons_per_hidden = 50
max_updates_per_env_step = 10
batch_size = 32
lrate = 1e-4

curr = Curriculum(curriculum=Curriculum.NO_CURRICULUM_VEL_1,
                  task=RobotEnv.TASK_REACH_CUBE,
                  max_steps_per_episode=max_steps_per_episode,
                  num_episodes=num_episodes,
                  num_hidden_layers=num_hidden_layers,
                  num_neurons_per_hidden=num_neurons_per_hidden,
                  batch_size=batch_size,
                  lrate=lrate,
                  testing_scripts=testing_scripts,  # ##
                  max_updates_per_env_step=max_updates_per_env_step,
                  )

no_curriculum_results_dict = curr.run()