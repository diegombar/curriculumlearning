from evaluate_curriculum import Curriculum

curr = Curriculum(curriculum=Curriculum.CURRICULUM_INCREASING_JOINT_NUMBER,
                  task=Curriculum.TASK_REACH_CUBE,
                  max_steps_per_episode=200,
                  num_episodes=5000,
                  num_hidden_layers=3,
                  num_neurons_per_hidden=50,
                  batch_size=32,
                  lrate=1e-4,
                  testing_scripts=False,  # ##
                  max_updates_per_env_step=10,
                  )

curr.run()
