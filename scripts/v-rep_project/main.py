
from evaluate_curriculum import Curriculum

curr = Curriculum(curriculum=Curriculum.NO_CURRICULUM_VEL_1,
                  task=Curriculum.TASK_REACH_CUBE,
                  max_steps_per_episode=200,
                  num_episodes=2000,
                  num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  batch_size=32,
                  lrate=1e-4,
                  testing_scripts=True,  # ##
                  max_updates_per_env_step=10,
                  )

curr.run()
