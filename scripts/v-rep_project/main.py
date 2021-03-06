from curriculum import Curriculum
from robotenv import RobotEnv

testing_scripts = True
curriculum = Curriculum.CURRICULUM_DECREASING_SPEED_SPARSE
# task = RobotEnv.TASK_REACH_CUBE
task = RobotEnv.TASK_REACH_CUBE
# max_steps_per_episode = 100
num_episodes = 2000  # estimate
max_steps_per_ep = 50  # estimate
max_total_transitions = num_episodes * max_steps_per_ep
num_hidden_layers = 3
num_neurons_per_hidden = 50
max_updates_per_env_step = 10
batch_size = 32
lrate = 1e-4
# replay_start_size = (num_episodes // 20) * max_steps_per_episode
# replay_memory_size = 10 * replay_start_size
disable_saving = True
sync_mode = True
portNb = 19999

curr_args = dict(curriculum=curriculum,
                 task=task,
                 # max_steps_per_episode=max_steps_per_episode,
                 # num_episodes=num_episodes,
                 max_total_transitions=max_total_transitions,
                 num_hidden_layers=num_hidden_layers,
                 num_neurons_per_hidden=num_neurons_per_hidden,
                 batch_size=batch_size,
                 lrate=lrate,
                 testing_scripts=testing_scripts,  # ##
                 max_updates_per_env_step=max_updates_per_env_step,
                 # replay_start_size=replay_start_size,
                 # replay_memory_size=replay_memory_size,
                 disable_saving=disable_saving,
                 sync_mode=sync_mode,
                 portNb=portNb,
                 )

curr = Curriculum(**curr_args)
no_curriculum_results_dict = curr.run()
