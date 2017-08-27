import time
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from robotenv import RobotEnv
from dql_algorithm import DQLAlgorithm


class Curriculum():
    # curriculums
    NO_CURRICULUM_SHAPING = 0
    NO_CURRICULUM_SPARSE = -2
    CURRICULUM_DECREASING_SPEED_SHAPING = 1
    CURRICULUM_DECREASING_SPEED_SPARSE = 4
    CURRICULUM_INCREASING_JOINT_NUMBER_SHAPING = 2
    CURRICULUM_INCREASING_JOINT_NUMBER_SPARSE = 5
    CURRICULUM_INITIALIZE_FURTHER_SHAPING = 3
    CURRICULUM_INITIALIZE_FURTHER_SPARSE = 6

    def __init__(self,
                 curriculum,
                 task,
                 # max_steps_per_episode,
                 # num_episodes,
                 max_total_transitions,
                 num_hidden_layers,
                 num_neurons_per_hidden,
                 batch_size,
                 lrate,
                 testing_scripts,
                 max_updates_per_env_step,
                 # replay_start_size=50000,
                 # replay_memory_size=500000,
                 disable_saving=False,
                 sync_mode=True,
                 portNb=19999,
                 ):
        self.curriculum = curriculum
        self.task = task
        # self.max_steps_per_episode = max_steps_per_episode
        # self.num_episodes = num_episodes
        self.max_total_transitions = max_total_transitions
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden = num_neurons_per_hidden
        self.batch_size = batch_size
        self.lrate = lrate
        self.testing_scripts = testing_scripts
        self.max_updates_per_env_step = max_updates_per_env_step
        # self.replay_start_size = (self.num_episodes // 20) * self.max_steps_per_episode
        # self.replay_memory_size = 10 * self.replay_start_size
        # self.replay_start_size = replay_start_size
        # self.replay_memory_size = replay_memory_size
        self.disable_saving = True if (disable_saving and testing_scripts) else False
        self.sync_mode = sync_mode
        self.portNb = portNb

        self.targetPosInitial = np.array([1.0] * 6) * np.pi
        # targetPosStraight = np.array([0.66, 1.0, 1.25, 1.5, 1.0, 1.0]) * np.pi
        self.targetPosHalfWayCube = np.array([0.66, 1.25, 1.25, 1.5, 1.0, 1.0]) * np.pi
        self.targetPosNearCube = np.array([0.66, 1.5, 1.25, 1.5, 1.0, 1.0]) * np.pi

        if self.task == RobotEnv.TASK_REACH_CUBE:
            self.task_name = 'reaching'
        elif self.task == RobotEnv.TASK_PUSH_CUBE_TO_TARGET_POSITION:
            self.task_name = 'pushing'
        else:
            raise RuntimeError('[CURRICULUM] Not a valid task.')

        # self.Velocities = [0.25]
        self.Velocities = np.array([1.0])
        self.NumOfAJoints = [6]
        self.Initial_positions = [self.targetPosInitial]

        if self.curriculum == self.NO_CURRICULUM_SHAPING or self.curriculum == self.NO_CURRICULUM_SPARSE:
            if self.curriculum == self.NO_CURRICULUM_SHAPING:
                self.curriculum_name = "no_cl_shaping"
            elif self.curriculum == self.NO_CURRICULUM_SPARSE:
                self.curriculum_name = "no_cl_sparse"
            self.Velocities = np.array([1.0])
        elif self.curriculum == self.CURRICULUM_DECREASING_SPEED_SHAPING or self.curriculum == self.CURRICULUM_DECREASING_SPEED_SPARSE:
            if self.curriculum == self.CURRICULUM_DECREASING_SPEED_SHAPING:
                self.curriculum_name = "cl_speeds_shaping"
            elif self.curriculum == self.CURRICULUM_DECREASING_SPEED_SPARSE:
                self.curriculum_name = "cl_speeds_sparse"
            self.Velocities = np.array([4.0, 2.0, 1.0])
            # self.replay_start_size = self.replay_start_size // len(self.Velocities)
            # self.num_episodes = self.num_episodes // len(self.Velocities)
            self.max_total_transitions = self.max_total_transitions // len(self.Velocities)
            # success_rate_for_subtask_completion = True
        elif self.curriculum == self.CURRICULUM_INCREASING_JOINT_NUMBER_SHAPING or self.curriculum == self.CURRICULUM_INCREASING_JOINT_NUMBER_SPARSE:
            if self.curriculum == self.CURRICULUM_INCREASING_JOINT_NUMBER_SHAPING:
                self.curriculum_name = "cl_joints_shaping"
            elif self.curriculum == self.CURRICULUM_INCREASING_JOINT_NUMBER_SPARSE:
                self.curriculum_name = "cl_joints_sparse"
            self.NumOfAJoints = range(1, 7)
            # self.replay_start_size = self.replay_start_size // len(self.NumOfAJoints)
            # self.num_episodes = self.num_episodes // len(self.NumOfAJoints)
            self.max_total_transitions = self.max_total_transitions // len(self.NumOfAJoints)
            # success_rate_for_subtask_completion = True
        elif self.curriculum == self.CURRICULUM_INITIALIZE_FURTHER_SHAPING or self.curriculum == self.CURRICULUM_INITIALIZE_FURTHER_SPARSE:
            if self.curriculum == self.CURRICULUM_INITIALIZE_FURTHER_SHAPING:
                self.curriculum_name = "cl_initial_states_shaping"
            elif self.curriculum == self.CURRICULUM_INITIALIZE_FURTHER_SPARSE:
                self.curriculum_name = "cl_initial_states_sparse"
            self.Initial_positions = [self.targetPosNearCube, self.targetPosHalfWayCube, self.targetPosInitial]
            # self.replay_start_size = self.replay_start_size // len(self.Initial_positions)
            # self.num_episodes = self.num_episodes // len(self.Initial_positions)
            self.max_total_transitions = self.max_total_transitions // len(self.Initial_positions)
        else:
            raise RuntimeError('[CURRICULUM] Not a valid curriculum.')

        if(self.curriculum == self.CURRICULUM_DECREASING_SPEED_SPARSE or
           self.curriculum == self.CURRICULUM_INCREASING_JOINT_NUMBER_SPARSE or
           self.curriculum == self.CURRICULUM_INITIALIZE_FURTHER_SPARSE or
           self.curriculum == self.NO_CURRICULUM_SPARSE
           ):
            # sparse rewards
            self.shaping_rewards = False
        else:
            # shaping rewards
            self.shaping_rewards = True

        if self.testing_scripts:
            self.curriculum_name += "_TEST"

        if not self.sync_mode:
            self.Velocities /= 4

        self.timestr = time.strftime("%b-%d_%H-%M-%S", time.gmtime())  # or time.localtime()
        self.current_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.all_curriculums_dir_path = os.path.join(self.current_dir_path, 'trained_models_and_results')
        self.folder_name = self.timestr + '_' + self.task_name + '_' + self.curriculum_name
        self.curriculum_dir_path = os.path.join(self.all_curriculums_dir_path, self.folder_name)
        self.serialized_lists_dir_path = os.path.join(self.curriculum_dir_path, 'serialized_curriculum_lists')

        dir_path_list = [self.all_curriculums_dir_path,
                         self.curriculum_dir_path,
                         self.serialized_lists_dir_path,
                         ]
        for new_directory in dir_path_list:
            os.makedirs(new_directory, exist_ok=True)

    def run(self):
        # vel025 = os.path.join(
        #    current_dir_path,"trained_models_and_results",
        #    "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")
        self.targetCubePosition = (0.15, 0.35)  # relative x, y in metres (robot base is at (0,0)), DOABLE
        subt_abs_initial_step = 0
        subt_abs_initial_episode = 0
        self.curriculum_switching_episodes = []
        self.curriculum_switching_steps = []
        self.curriculum_num_steps_per_ep = []
        self.curriculum_undisc_return_per_ep = []
        self.curriculum_epsilon_per_ep = []
        self.curriculum_net_updates_per_step = []
        self.curriculum_success_step_per_ep = []
        self.curriculum_test_success_rates = [0.0]
        self.curriculum_test_mean_returns = [0.0]
        self.curriculum_total_time = 0.0
        cl_cumul_successes_per_ep = np.array([])
        cl_test_steps = np.array([0])
        cl_test_episodes = np.array([0])

        subt_trained_model_save_path = None

        trainDQL_args = dict(experiment_dir_path=self.curriculum_dir_path,
                             num_hidden_layers=self.num_hidden_layers,
                             num_neurons_per_hidden=self.num_neurons_per_hidden,
                             # num_episodes=self.num_episodes,  # 400
                             max_total_transitions=self.max_total_transitions,
                             # max_steps_per_episode=self.max_steps_per_episode,  # 200
                             e_min=0.01,
                             task=self.task,
                             lrate=self.lrate,  # 1e-3 seems to work fine
                             batch_size=self.batch_size,
                             # replay_start_size=self.replay_start_size,
                             # replay_memory_size=self.replay_memory_size,
                             showGUI=True,
                             # velocity=0.25,  # 1.0 seems to work fine
                             model_to_load_file_path=subt_trained_model_save_path,
                             use_variable_names=True,
                             skip_training=False,
                             notes=self.curriculum_name,
                             previous_norm=False,
                             targetCubePosition=self.targetCubePosition,
                             # policy_test_period=50,  # episodes
                             # policy_test_episodes=20,  # episodes
                             # success_rate_for_subtask_completion=success_rate_for_subtask_completion,  # change with/without CL
                             nSJoints=6,
                             # nAJoints=6,
                             portNb=self.portNb,
                             old_bias=False,
                             max_updates_per_env_step=self.max_updates_per_env_step,
                             disable_saving=self.disable_saving,
                             sync_mode=self.sync_mode,
                             shaping_rewards=self.shaping_rewards,
                             )

        if self.testing_scripts:
            trainDQL_args.update(dict(batch_size=1,
                                      # max_total_transitions=self.max_total_transitions,
                                      # num_episodes=5,
                                      # max_steps_per_episode=2,
                                      # replay_start_size=6,
                                      # replay_memory_size=10,
                                      policy_test_period=5,  # episodes
                                      policy_test_episodes=2,  # episodes
                                      ))

        # run curriculum
        print('\n[CURRICULUM] Running new curriculum: ', self.curriculum_name)
        st_num = 0
        for vel in self.Velocities:
            for nAJoints in self.NumOfAJoints:
                for initial_joint_positions in self.Initial_positions:
                    max_steps_per_episode = 0
                    # adapt number of steps: following values testes with real vel = 1.0
                    if ((self.task == RobotEnv.TASK_REACH_CUBE or
                         self.task == RobotEnv.TASK_PUSH_CUBE_TO_TARGET_POSITION
                         )):
                        test_max_steps_per_episode = 50
                        if np.array_equal(initial_joint_positions, self.targetPosNearCube):
                            max_steps_per_episode = 5
                        elif np.array_equal(initial_joint_positions, self.targetPosHalfWayCube):
                            max_steps_per_episode = 20
                        elif np.array_equal(initial_joint_positions, self.targetPosInitial):
                            max_steps_per_episode = test_max_steps_per_episode
                    if self.task == RobotEnv.TASK_PUSH_CUBE_TO_TARGET_POSITION:
                        extra_steps_to_push = 15
                        max_steps_per_episode += extra_steps_to_push
                        test_max_steps_per_episode += extra_steps_to_push
                    replay_start_size = max((self.num_episodes // 20), 3) * max_steps_per_episode
                    replay_memory_size = 10 * replay_start_size

                    if self.testing_scripts:
                        self.max_total_transitions = 5 * max_steps_per_episode
                        trainDQL_args.update(dict(max_total_transitions=self.max_total_transitions))
                    # self.max_steps_per_episode
                    st_num += 1
                    trainDQL_args.update(dict(velocity=vel,
                                              nAJoints=nAJoints,
                                              initial_joint_positions=initial_joint_positions,
                                              model_to_load_file_path=subt_trained_model_save_path,
                                              max_steps_per_episode=max_steps_per_episode,
                                              replay_start_size=replay_start_size,
                                              replay_memory_size=replay_memory_size,
                                              test_max_steps_per_episode=test_max_steps_per_episode,
                                              )
                                         )

                    # model_path, subt_total_steps, subt_cumul_successes_per_ep, subt_test_success_rates, subt_test_steps, subt_total_training_time_in_hours = training.trainDQL(**trainDQL_args)

                    dql = DQLAlgorithm(**trainDQL_args)

                    # model_path, subt_total_steps, subt_cumul_successes_per_ep, subt_total_training_time_in_hours = dql.run()
                    results_dict = dql.run()

                    subt_undisc_return_per_ep = results_dict['undisc_return_per_ep']
                    subt_num_steps_per_ep = results_dict['num_steps_per_ep']
                    subt_cumul_successes_per_ep = results_dict['cumul_successes_per_ep']
                    subt_epsilon_per_ep = results_dict['epsilon_per_ep']
                    subt_success_step_per_ep = results_dict['success_step_per_ep']
                    subt_test_steps = results_dict['test_steps']
                    subt_test_episodes = results_dict['test_episodes']
                    subt_test_success_rates = results_dict['test_success_rates']
                    subt_test_mean_returns = results_dict['test_mean_returns']
                    subt_net_updates_per_step = results_dict['net_updates_per_step']
                    subt_trained_model_save_path = results_dict['trained_model_save_path']
                    subt_total_steps = results_dict['total_steps']
                    subt_total_episodes = results_dict['total_episodes']
                    subt_total_training_time_in_hours = results_dict['total_training_time_in_hours']

                    # update cumulative successes
                    last_cumul_successes_per_step = 0 if st_num == 1 else cl_cumul_successes_per_ep[-1]
                    abs_subt_cumul_successes_per_ep = np.array(subt_cumul_successes_per_ep) + last_cumul_successes_per_step  # subtask counter to cl counter
                    cl_cumul_successes_per_ep = np.concatenate((cl_cumul_successes_per_ep, abs_subt_cumul_successes_per_ep), axis=0)

                    # update test steps and test episodes
                    abs_subt_test_steps = np.array(subt_test_steps) + subt_abs_initial_step  # subtask step to cl step
                    cl_test_steps = np.concatenate((cl_test_steps, abs_subt_test_steps), axis=0)
                    abs_subt_test_episodes = np.array(subt_test_episodes) + subt_abs_initial_episode  # subtask episode to cl episode
                    cl_test_episodes = np.concatenate((cl_test_episodes, abs_subt_test_episodes), axis=0)

                    # update switching steps and switching episodes
                    last_curriculum_step = subt_total_steps + subt_abs_initial_step
                    self.curriculum_switching_steps.append(last_curriculum_step)
                    last_curriculum_ep = subt_total_episodes + subt_abs_initial_episode
                    self.curriculum_switching_episodes.append(last_curriculum_ep)

                    # list concatenations
                    self.curriculum_num_steps_per_ep += subt_num_steps_per_ep
                    self.curriculum_undisc_return_per_ep += subt_undisc_return_per_ep
                    self.curriculum_epsilon_per_ep += subt_epsilon_per_ep
                    self.curriculum_success_step_per_ep += subt_success_step_per_ep
                    self.curriculum_net_updates_per_step += subt_net_updates_per_step
                    self.curriculum_test_success_rates += subt_test_success_rates
                    self.curriculum_test_mean_returns += subt_test_mean_returns

                    # update curriculum time
                    self.curriculum_total_time += subt_total_training_time_in_hours  # in hours
                    subt_stats_dict = dict(number_of_steps_executed_curriculum_so_far=last_curriculum_step,
                                           number_of_episodes_executed_curriculum_so_far=last_curriculum_ep,
                                           training_time_in_hours_curriculum_so_far=self.curriculum_total_time,
                                           )
                    stats_file_path = os.path.join(self.curriculum_dir_path, "stats_sub_task_" + str(st_num) + ".txt")
                    with open(stats_file_path, "w") as stats_file:
                        json.dump(subt_stats_dict, stats_file, sort_keys=True, indent=4)

                    subt_abs_initial_step = last_curriculum_step
                    subt_abs_initial_episode = last_curriculum_ep

        # curriculum ended, serializing curriculum lists
        curriculum_end_stats_dict = dict(total_number_of_steps_executed_curriculum=last_curriculum_step,
                                         total_number_of_episodes_executed_curriculum=last_curriculum_ep,
                                         total_training_time_in_hours_curriculum=self.curriculum_total_time
                                         )
        end_stats_file_path = os.path.join(self.curriculum_dir_path, "end_stats.txt")
        with open(end_stats_file_path, "w") as end_stats_file:
            json.dump(curriculum_end_stats_dict, end_stats_file, sort_keys=True, indent=4)

        # save lists of results for later plots
        self.curriculum_cumul_successes_per_ep = cl_cumul_successes_per_ep.tolist()
        self.curriculum_test_steps = cl_test_steps.tolist()
        self.curriculum_test_episodes = cl_test_episodes.tolist()

        episodes = range(1, len(self.curriculum_undisc_return_per_ep) + 1)
        steps = range(1, len(self.curriculum_net_updates_per_step) + 1)

        cumul_cl_steps = np.cumsum(np.array(self.curriculum_num_steps_per_ep))  # 1 x num episodes

        self.savePlot(self.curriculum_dir_path,
                      'episodes', episodes,
                      'steps', self.curriculum_num_steps_per_ep,
                      'Environment steps during training',
                      'curriculum_num_steps_per_ep',
                      self.curriculum_switching_episodes,
                      )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', cumul_cl_steps,
                      'returns', self.curriculum_undisc_return_per_ep,
                      'Undiscounted returns during training',
                      'curriculum_undisc_return_by_transitions',
                      self.curriculum_switching_steps,
                      )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', cumul_cl_steps,
                      'successful episodes', self.curriculum_cumul_successes_per_ep,
                      'Cumulative successful episodes during training',
                      'curriculum_cumul_successes_by_transition',
                      self.curriculum_switching_steps,
                      )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', cumul_cl_steps,
                      'epsilon', self.curriculum_epsilon_per_ep,
                      'Epsilon evolution during training',
                      'curriculum_epsilon_by_transitions',
                      self.curriculum_switching_steps,
                      )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', cumul_cl_steps,
                      'steps', self.curriculum_success_step_per_ep,
                      'Steps to reach a goal state during training',
                      'curriculum_success_step_by_transitions',
                      self.curriculum_switching_steps,
                      )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', self.curriculum_test_steps,
                      'success rate', self.curriculum_test_success_rates,
                      'Success rate in test conditions',
                      'curriculum_success_rate_by_transitions',
                      self.curriculum_switching_steps,
                      )

        # self.savePlot(self.curriculum_dir_path,
        #               'episodes', self.curriculum_test_episodes,
        #               'success rate', self.curriculum_test_success_rates,
        #               'Success rate in test conditions',
        #               'curriculum_success_rate_per_ep',
        #               self.curriculum_switching_episodes,
        #               )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', self.curriculum_test_steps,
                      'mean return', self.curriculum_test_mean_returns,
                      'Mean undisc. return in test conditions',
                      'curriculum_test_mean_returns_by_transitions',
                      self.curriculum_switching_steps,
                      )

        # self.savePlot(self.curriculum_dir_path,
        #               'episodes', self.curriculum_test_episodes,
        #               'mean return', self.curriculum_test_mean_returns,
        #               'Mean undisc. return in test conditions',
        #               'curriculum_test_mean_returns_per_ep',
        #               self.curriculum_switching_episodes,
        #               )

        self.savePlot(self.curriculum_dir_path,
                      'transitions', steps,
                      'updates', self.curriculum_net_updates_per_step,
                      'Number of network updates during training',
                      'curriculum_net_updates_per_step',
                      self.curriculum_switching_steps,
                      )

        lists_to_serialize_dict = dict(curriculum_undisc_return_per_ep=self.curriculum_undisc_return_per_ep,
                                       curriculum_num_steps_per_ep=self.curriculum_num_steps_per_ep,
                                       curriculum_cumul_successes_per_ep=self.curriculum_cumul_successes_per_ep,
                                       curriculum_epsilon_per_ep=self.curriculum_epsilon_per_ep,
                                       curriculum_success_step_per_ep=self.curriculum_success_step_per_ep,
                                       curriculum_test_steps=self.curriculum_test_steps,
                                       curriculum_test_episodes=self.curriculum_test_episodes,
                                       curriculum_test_success_rates=self.curriculum_test_success_rates,
                                       curriculum_test_mean_returns=self.curriculum_test_mean_returns,
                                       curriculum_net_updates_per_step=self.curriculum_net_updates_per_step,
                                       curriculum_switching_episodes=self.curriculum_switching_episodes,
                                       curriculum_switching_steps=self.curriculum_switching_steps,
                                       )

        self.serialize_lists(self.serialized_lists_dir_path, lists_to_serialize_dict)

        return lists_to_serialize_dict

    def serialize_lists(self, dir_path, dict_of_lists):
        for list_name, list_values in dict_of_lists.items():
            list_json_file = os.path.join(dir_path, list_name + '.json')
            with open(list_json_file, "w") as json_file:
                list_var_name = 'self.' + list_name
                json.dump(eval(list_var_name), json_file)

    def savePlot(self,
                 dir_path,
                 x_label, x_values,
                 y_label, y_values,
                 title,
                 filename,
                 vertical_xs=None
                 ):
        fig = plt.figure()
        # start at episode 1
        plt.plot(x_values, y_values, linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if vertical_xs is not None:
            # vertical lines at subtask switching
            for vertical_x in vertical_xs:
                plt.axvline(x=vertical_x, color='g', ls=':', linewidth=0.5)
        plot_file = os.path.join(dir_path, filename + '.svg')
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close()
