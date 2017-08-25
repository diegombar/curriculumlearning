from evaluate_curriculum import Curriculum
from robotenv import RobotEnv

testing_scripts = True
curriculum = Curriculum.CURRICULUM_INITIALIZE_FURTHER
task = RobotEnv.TASK_REACH_CUBE
max_steps_per_episode = 200
num_episodes = 2000
num_hidden_layers = 2
num_neurons_per_hidden = 50
max_updates_per_env_step = 10
batch_size = 32
lrate = 1e-4
replay_start_size = (num_episodes // 20) * max_steps_per_episode
replay_memory_size = 10 * replay_start_size
disable_saving = True

curr_args = dict(curriculum=curriculum,
                 task=task,
                 max_steps_per_episode=max_steps_per_episode,
                 num_episodes=num_episodes,
                 num_hidden_layers=num_hidden_layers,
                 num_neurons_per_hidden=num_neurons_per_hidden,
                 batch_size=batch_size,
                 lrate=lrate,
                 testing_scripts=testing_scripts,  # ##
                 max_updates_per_env_step=max_updates_per_env_step,
                 replay_start_size=replay_start_size,
                 replay_memory_size=replay_memory_size,
                 disable_saving=disable_saving,
                 )

curr = Curriculum(**curr_args)
no_curriculum_results_dict = curr.run()
