# coding: utf-8
import os
import numpy as np
# import random
import tensorflow as tf
from matplotlib import pyplot as plt
# import readchar
import time
import random
import os.path
import json
import subprocess
import socket
# import logging
import threading
# Load environment
from robotenv import RobotEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(levelname)s] (%(threadName)-10s) %(message)s',
#                     )


# experience replay dataset, experience = (s,a,r,s',done)
class experience_dataset():

    def __init__(self, size):
        self.data = []
        self.size = size
        self.datalock = threading.Lock()

    #add experience = list of transitions, freeing space if needed
    def add(self, experience):
        with self.datalock:
            excess = len(self.data) + len(experience) - self.size
            if excess > 0:
                self.data[0:excess] = []
            self.data.extend(experience)

    # randomly sample an array of transitions (s,a,r,s',done)
    def sample(self, sample_size):
        with self.datalock:
            sample = np.array(random.sample(self.data,sample_size))
        return np.reshape(sample, [sample_size, 5])

class DQN():
    def __init__(self,
        nAJoints,
        nSJoints,
        stateSize,
        num_hidden_layers,
        num_neurons_per_hidden,
        learning_rate,
        use_variable_names=True,
        previous_norm=False,
        old_bias=False,
        add_summary=False,
        ):
        self.nAJoints = nAJoints
        self.nSJoints = nSJoints
        self.stateSize = stateSize
        self.nActionsPerJoint = 3
        self.nActions = self.nAJoints * self.nActionsPerJoint
        self.inState = tf.placeholder(shape=[None, self.stateSize], dtype=tf.float32, name='state')  #batch_size x stateSize
        self.add_summary = add_summary
        self.variable_dict = {}

        # architecture

        nHidden = num_neurons_per_hidden
        # list of layer sizes
        neuronsPerLayer = [num_neurons_per_hidden] * (num_hidden_layers + 2)
        
        if use_variable_names:
            neuronsPerLayer[0] = 30  # large state for CL, weights are adapted to stateSize below
            neuronsPerLayer[-1] = 6 * self.nActionsPerJoint  # same as above for actions
            # initialize params
            self.weight_names = []
            self.bias_names = []
            self.hidden_layers = []
            for i in range(len(neuronsPerLayer) - 1):
                layer_name = 'layer' + str(i)
                weight_name = "weight" + str(i)
                bias_name = "bias" + str(i)
                with tf.name_scope(layer_name) as scope:
                    self.variable_dict[weight_name] = tf.Variable(tf.truncated_normal([neuronsPerLayer[i], neuronsPerLayer[i+1]], mean=0.0, stddev=0.1), name=weight_name)
                    bias_shape = [1] if old_bias else [neuronsPerLayer[i+1]]
                    self.variable_dict[bias_name] = tf.Variable(tf.constant(0.1, shape=bias_shape), name=bias_name)
                    self.weight_names.append(weight_name)
                    self.bias_names.append(bias_name)
                    if i == 0:
                        # ignore some weights to adapt to the stateSize (for CL)
                        weight_input = tf.slice(self.variable_dict[weight_name], [0, 0], [self.stateSize, -1], name='weight_input')
                        input_multiplier = 2 if previous_norm else 1
                        first_layer_input = input_multiplier * self.inState
                        self.hidden_layers.append(tf.nn.relu(tf.matmul(first_layer_input, weight_input) + self.variable_dict[bias_name], name="neuron_activations" + str(i)))
                        self.variable_summaries('activations', self.hidden_layers[-1])
                        # tf.summary.histogram('activations', self.hidden_layers[-1])
                    elif i < (len(neuronsPerLayer) - 2):
                        self.hidden_layers.append(tf.nn.relu(tf.matmul(self.hidden_layers[-1], self.variable_dict[weight_name]) + self.variable_dict[bias_name], name="neuron_activations" + str(i)))
                        self.variable_summaries('activations', self.hidden_layers[-1])
                        # tf.summary.histogram('activations', self.hidden_layers[-1])
                    else:
                        # last layer
                        weight_output = tf.slice(self.variable_dict[weight_name], [0, 0], [-1, self.nActions], name='weight_output')
                        bias_output = tf.slice(self.variable_dict[bias_name], [0], [self.nActions], name='bias_output')
                        self.allJointsQvalues = tf.add(tf.matmul(self.hidden_layers[-1], weight_output), bias_output, name="q_values")  # Q values for all actions given inState, #batch_size x nActions
                        self.variable_summaries('activations', self.allJointsQvalues)
                        # tf.summary.histogram('activations', self.allJointsQvalues)
                    # summaries
                    self.variable_summaries('weights', self.variable_dict[weight_name])
                    self.variable_summaries('biases', self.variable_dict[bias_name])

            # self.variable_dict = {
            #                 "weight0":self.weights[0],
            #                 "weight1":self.weights[1],
            #                 "weight1":self.weights[2],
            #                 "bias0":self.biases[0],
            #                 "bias1":self.biases[1],
            #                 "bias2":self.biases[2],
            #             }
        else:
            # for visualizarion of first few tained models
            neuronsPerLayer[0] = 6
            neuronsPerLayer[-1] = 6 * self.nActionsPerJoint
            self.variable_dict = {}
            VariableNames = ["Variable", "Variable_1", "Variable_2", "Variable_3", "Variable_4 ","Variable_5"]
            N = len(neuronsPerLayer) - 1
            for i in range(N):
                weight_name = "weight" + str(i)
                bias_name = "bias" + str(i)
                self.variable_dict[VariableNames[i]] = tf.Variable(tf.truncated_normal([neuronsPerLayer[i], neuronsPerLayer[i+1]], mean=0.0, stddev=0.1),
                                  # name=weight_name
                                  )
                self.variable_dict[VariableNames[i+N]] = tf.Variable(tf.constant(0.1, shape=[1]),  # should be shape=[nHidden]
                                  # name=bias_name
                                  )
            # layers
            input_multiplier = 2 if previous_norm else 1
            first_layer_input = input_multiplier * self.inState
            self.hidden1 = tf.nn.relu(tf.matmul(first_layer_input, self.W0) + self.b0)
            self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W1) + self.b1)
            self.allJointsQvalues = add(tf.matmul(self.hidden2, self.W2), self.b2) # Q values for all actions given inState, #batch_size x nActions

        # Training

        # get actions of highest Q value for each joint
        with tf.name_scope('indep_joints') as scope:
            self.allJointsQvalues3D = tf.reshape(self.allJointsQvalues, [-1, self.nAJoints, self.nActionsPerJoint]) # batch_size x nJoints x actionsPerJoint
            self.allJointsBestActions = tf.argmax(self.allJointsQvalues3D, axis=2, name='best_actions') # batch_size x nJoints

        # get Q target values
        self.Qtargets = tf.placeholder(shape=[None, self.nAJoints], dtype=tf.float32, name='q_targets') #batch_size x nJoints
        # get batch of executed actions a0
        self.chosenActions = tf.placeholder(shape=[None, self.nAJoints],dtype=tf.int32, name='chosen_actions') #batch_size x nJoints

        #get Q values corresponding to executed actions (i.e. Q values to update)
        with tf.name_scope('q_values_to_update') as scope:
            self.chosenAs_onehot = tf.one_hot(self.chosenActions, self.nActionsPerJoint, dtype=tf.float32) #batch_size x nJoints x nActionsPerJoint
            self.chosenActionsQvalues = tf.reduce_sum(tf.multiply(self.allJointsQvalues3D, self.chosenAs_onehot), axis=2, name='executed_actions_q_values') #element-wise multiplication

        # loss by taking the sum of squares difference between the target and predicted Q values
        with tf.name_scope('training') as scope:
            self.error = tf.square(self.Qtargets - self.chosenActionsQvalues, name='error') #element-wise
            self.loss = tf.reduce_sum(self.error, name='loss')
            if self.add_summary:
                tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.updateModel = self.optimizer.minimize(self.loss)

    def variable_summaries(self, name, var):
        if self.add_summary:
            """TensorBoard visualization"""
            with tf.name_scope(name + '_summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)



class DQLAlgorithm():
    def __init__(self,
        experiment_dir_path,
        num_hidden_layers, num_neurons_per_hidden,
        num_episodes, max_steps_per_episode, e_min,
        task,
        model_saving_period=100,
        lrate=1E-6,  #scripts/v-rep_project/training_independent_joints.py
        batch_size=32,
        replay_start_size=50000,
        replay_memory_size=500000,
        showGUI=True,
        velocity=0.3,
        model_to_load_file_path=None,
        use_variable_names=True,
        skip_training=False,
        notes=None,
        previous_norm=False,
        targetRelativePos=0,
        # policy_test_period=100,  # episodes
        # success_rate_for_subtask_completion=False,
        nSJoints=6,
        nAJoints=6,
        portNb=19998,
        old_bias=False,
        # max_updates_per_env_step=10
        ):

        self.experiment_dir_path = experiment_dir_path
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden = num_neurons_per_hidden
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.e_min = e_min
        self.task = task
        self.model_saving_period = model_saving_period
        self.lrate = lrate
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.replay_memory_size = replay_memory_size
        self.showGUI = showGUI
        self.velocity = velocity
        self.model_to_load_file_path = model_to_load_file_path
        self.use_variable_names = use_variable_names
        self.skip_training = skip_training
        self.notes = notes
        self.previous_norm = previous_norm
        self.targetRelativePos = targetRelativePos
        # self.policy_test_period = policy_test_period
        # self.success_rate_for_subtask_completion = success_rate_for_subtask_completion
        self.nSJoints = nSJoints
        self.nAJoints = nAJoints
        self.portNb = portNb
        self.old_bias = old_bias
        # self.max_updates_per_env_step = max_updates_per_env_step

        self.h_params = {}
        self.end_stats_dict = {}

        # hyper params to save to txt file
        self.h_params["experiment_dir_path"] = self.experiment_dir_path
        self.h_params["showGUI"] = self.showGUI
        self.h_params['num_hidden_layers'] = self.num_hidden_layers # not counting output layer
        self.h_params['neurons_per_hidden_layer'] = self.num_neurons_per_hidden  #mnih: 512 for dense hidden layer
        self.h_params['num_episodes'] = self.num_episodes
        self.h_params['max_steps_per_episode'] = self.max_steps_per_episode
        self.h_params['model_saving_period'] = self.model_saving_period #in eps
        self.h_params['q_plots_period'] = self.q_plots_period = max(self.model_saving_period//5, 1) #in eps
        self.h_params['q_plots_num_of_points'] = self.q_plots_num_of_points = max(self.q_plots_period//4, 1) #in eps
        self.h_params['tensorboard_log_period'] = self.tensorboard_log_period = max(self.model_saving_period//10, 1) * self.max_steps_per_episode * 10 # in network updates ~ env steps
        self.h_params['e_min'] = self.e_min
        self.h_params['task'] = self.task
        self.h_params['batch_size'] = self.batch_size #mnih=32
        self.h_params['replay_start_size'] = self.replay_start_size # steps to fill dataset with random actions mnih=5E4
        self.h_params['replay_memory_size'] = self.replay_memory_size # in steps #mnih: 1E6
        self.h_params["joint_velocity"] = self.velocity
        self.h_params['use_variable_names'] = self.use_variable_names #use non-default names for variables

        # self.h_params['policy_test_period'] = policy_test_period
        # self.h_params['success_rate_for_subtask_completion'] = success_rate_for_subtask_completion

        self.h_params['number_of_state_joints'] = nSJoints
        self.h_params['number_of_action_joints'] = nAJoints

        self.h_params['hostname'] = socket.gethostname()

        # load model if path is specified
        self.load_model = False
        if self.model_to_load_file_path is not None:
            # e.g.
            # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","saved_checkpoints","checkpoint_model-400")
            # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","trained_model","final_model-2000")
            self.h_params["model_to_load_file_path"] = self.model_to_load_file_path
            self.load_model = True

        self.h_params['load_model'] = self.load_model
        self.h_params['skip_training'] = self.skip_training #skip for visualization, do not skip for curriculum learning/pre-training

        if self.notes is not None: self.h_params['notes'] = self.notes


        if self.replay_start_size <= self.max_steps_per_episode or self.replay_start_size < self.batch_size:
            print("WARNING: replay_start_size must be greater than max_steps_per_episode and batch_size")

        # create folders to save results
        # current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
        # all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
        # experiment_dir_path = os.path.join(all_models_dir_path, experiment_folder_name)
        timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
        self.current_model_dir_path = os.path.join(self.experiment_dir_path, "model_and_results_" + timestr)
        self.trained_model_plots_dir_path = os.path.join(self.current_model_dir_path, "trained_model_results")
        self.checkpoints_dir_path = os.path.join(self.current_model_dir_path, "saved_checkpoints")
        self.trained_model_dir_path = os.path.join(self.current_model_dir_path, "trained_model")
        self.checkpoint_model_file_path = os.path.join(self.checkpoints_dir_path, "checkpoint_model")
        self.trained_model_file_path = os.path.join(self.trained_model_dir_path, "final_model")
        self.log_file_path = os.path.join(self.current_model_dir_path,"logs")

        for new_directory in [self.trained_model_plots_dir_path, self.checkpoints_dir_path, self.trained_model_dir_path, self.log_file_path]:
            os.makedirs(new_directory, exist_ok=True)

        # save git commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        self.h_params["_commit_hash"] = git_hash.decode("utf-8").strip()

        #plot and save reward function
        self.h_params['rewards_normalizer'] = self.rewards_normalizer = 0.1
        distanceOfRewardCloseToZero = 1.0
        self.h_params['rewards_decay_rate'] = self.rewards_decay_rate = 1.0/ (distanceOfRewardCloseToZero / 5) #=1/0.33 i.e. near 0 at 5 * 0.33 = 1.65m away

        # recursive exponential decay for epsilon
        self.h_params['e_max'] = self.e_max = 1.0 #P(random action in at least one joint) = 1- (1 - epsilon)**nJoints
        self.h_params['e_tau'] = e_tau = (self.num_episodes - (self.replay_start_size // self.max_steps_per_episode)) / 5 # time constant in episodes, close to final value at 5 e_tau
        self.addEFactor = 1.0 - (1.0 / e_tau)

        self.h_params['train_model_steps_period'] = train_model_steps_period = 4 # mnih = 4, period of mini-batch sampling and training
        self.h_params['update_target_net_rate_tau'] = self.tau = 0.001 # rate to update target network toward main network
        self.h_params['learning_rate'] = lrate #= 1E-6
        self.h_params['discount_factor'] = self.y = 0.99 # mnih:0.99
        # self.h_params['policy_test_period'] = policy_test_period #episodes
        # self.h_params['policy_test_num_of_test_episodes '] = policy_test_episodes = 20 # episodes

        self.dataset = experience_dataset(self.replay_memory_size)
        self.graphlock = threading.Lock()


    def get_graphs_ops(self):
        total_vars = len(self.trainables)
        one_net_vars_length = total_vars // 3
        trainer_target_soft_op_holder = []
        trainer_target_copy_op_holder = []
        collector_main_op_holder = []

        # trainables thirds: 1st= trainer_main_net, 2nd=trainer_target_net, 3rd=collector_main_net
        for idx, trainer_main_net_var in enumerate(self.trainables[0:one_net_vars_length]):
            trainer_target_net_var = self.trainables[idx + one_net_vars_length]
            collector_main_net_var = self.trainables[idx + 2 * one_net_vars_length]
            trainer_target_soft_op = trainer_target_net_var.assign(self.tau * trainer_main_net_var.value() + (1 - self.tau)*trainer_target_net_var.value())
            trainer_target_copy_op = trainer_target_net_var.assign(trainer_main_net_var.value())
            collector_main_update_op = collector_main_net_var.assign(trainer_main_net_var.value())
            trainer_target_soft_op_holder.append(trainer_target_soft_op)
            trainer_target_copy_op_holder.append(trainer_target_copy_op)
            collector_main_op_holder.append(collector_main_update_op)
        return trainer_target_soft_op_holder, collector_main_op_holder, trainer_target_copy_op_holder


    def updateCollectorNetParams(self,sess):
        self.run_ops(self.collector_main_ops,sess)


    def copyLearnerMainToTarget(self,sess):
        self.run_ops(self.trainer_target_copy_ops,sess)


    def softUpdateTarget(self,sess):
        self.run_ops(self.trainer_target_soft_ops,sess)


    def run_ops(self, op_holder, sess):
        with self.graphlock:
            for op in op_holder:
                sess.run(op)


    def run(self):
        with RobotEnv(task=self.task,
                      targetPosition=self.targetRelativePos,
                      rewards_normalizer=self.rewards_normalizer,
                      rewards_decay_rate=self.rewards_decay_rate,
                      showGUI=self.showGUI,
                      velocity=self.velocity,
                      nSJoints=self.nSJoints,
                      nAJoints=self.nAJoints,
                      portNb=self.portNb
                      ) as env:
            tf.reset_default_graph()

            self.nActionsPerJoint = 3
            self.h_params['state_size'] = self.stateSize = env.observation_space_size
            self.h_params['number_of_actions'] = nActions = env.action_space_size

            self.saveRewardFunction(env, self.current_model_dir_path)

            dqn_params = dict(
                            nAJoints=self.nAJoints,
                            nSJoints=self.nSJoints,
                            stateSize=self.stateSize,
                            num_hidden_layers=self.num_hidden_layers,
                            num_neurons_per_hidden=self.num_neurons_per_hidden,
                            learning_rate=self.lrate,
                            use_variable_names=self.use_variable_names,
                            previous_norm=self.previous_norm,
                            old_bias=self.old_bias,
                            add_summary=False,
                            )

            main_dqn_params = dqn_params.copy()
            main_dqn_params.update(dict(add_summary=True))
            self.trainer_mainDQN = DQN(**main_dqn_params)

            self.trainer_targetDQN = DQN(**dqn_params)

            self.collector_mainDQN = DQN(**dqn_params)

            # save txt file with hyper parameters
            h_params_file_path = os.path.join(self.current_model_dir_path,
                                              "hyper_params.txt")
            with open(h_params_file_path, "w") as h_params_file:
                json.dump(self.h_params, h_params_file, sort_keys=True, indent=4)

            # initialize and create variables saver
            init = tf.global_variables_initializer()

            self.coord = tf.train.Coordinator()

            # variable_dict = mainDQN.variable_dict
            # variable_dict = {
            #                     "weight0":mainDQN.weights[0],
            #                     "weight1":mainDQN.weights[1],
            #                     "weight1":mainDQN.weights[2],
            #                     "bias0":mainDQN.biases[0],
            #                     "bias1":mainDQN.biases[1],
            #                     "bias2":mainDQN.biases[2],
            #                 }

            # if not use_variable_names:
            #     variable_dict = {
            #                         "Variable":mainDQN.W0,
            #                         "Variable_1":mainDQN.W1,
            #                         "Variable_2":mainDQN.W2,
            #                         "Variable_3":mainDQN.b0,
            #                         "Variable_4":mainDQN.b1,
            #                         "Variable_5":mainDQN.b2,
            #                     }

            saver = tf.train.Saver(self.trainer_mainDQN.variable_dict)
            self.trainables = tf.trainable_variables()
            self.trainer_target_soft_ops, self.collector_main_ops, self.trainer_target_copy_ops = self.get_graphs_ops()

            with tf.Session() as sess:                
                start_time = time.time()
                self.merged = tf.summary.merge_all()
                self.training_writer = tf.summary.FileWriter(self.log_file_path + '/training', sess.graph) #TensorBoard
                sess.run(init)

                if self.load_model:
                    print('\n################ LOADING SAVED MODEL ###############')
                    saver.restore(sess, self.model_to_load_file_path)
                    print('\n######## SAVED MODEL WAS SUCCESSFULLY LOADED #######')

                self.subtask_total_steps = 0

                if not self.skip_training:
                    trainer_thread = threading.Thread(name='trainer', target=self.trainer, args=(sess,))
                    print("\nTrainer thread created.")
                    trainer_thread.start()


                self.copyLearnerMainToTarget(sess)  # Set the target network to be equal to the primary network.
                # self.run_ops(self.trainer_target_ops, sess)  #soft update
                # initialize epsilon
                epsilon = self.e_min
                if not self.skip_training:
                    addE = self.e_max - self.e_min
                    epsilon = self.e_min + addE

                self.num_steps_per_ep = []
                self.undisc_return_per_ep = []
                success_count = 0
                self.subt_cumul_successes = []
                self.epsilon_per_ep = []
                self.average_maxQ_per_ep = np.array([]).reshape(self.nAJoints, 0)
                self.statesArray = np.array([]).reshape(self.stateSize, 0)
                self.maxQvaluesArray = np.array([]).reshape(self.nAJoints, 0)
                self.success_steps = []
                self.is_saving = False
                total_saving_time = 0

                # test success initialization:
                # subt_test_success_rates = []
                # subt_test_steps = []
                # testing_policy_episode = 0
                # testing_policy = False
                # epsilon_backup = 0
                # no_progress_count = 0
                # current_test_success_count = 0

                collector_start_time = time.time()
                self.current_episode = 1
                while self.current_episode <= self.num_episodes:
                    self.updateCollectorNetParams(sess) #copy the trainer main net to the collector main net
                    # if testing_policy:
                    #     print("\nTesting episode number ", testing_policy_episode)
                    # else:
                    print("\nTraining episode number ", self.current_episode)
                    task_completed = False

                    if self.skip_training:
                        epsilon = self.e_min
                    elif self.subtask_total_steps > self.replay_start_size:
                        # decay epsilon
                        addE *= self.addEFactor
                        epsilon = self.e_min + addE

                    initialState = env.reset() # reset environment and get first observation
                    # print("\ninitialState: ", initialState)
                    undisc_return = 0
                    sum_of_maxQ = np.zeros((self.nAJoints, 1))
                    done = False
                    if not self.skip_training: episodeBuffer = experience_dataset(self.replay_memory_size) # temporary buffer

                    self.current_step = 1
                    while self.current_step <= self.max_steps_per_episode:
                        print("\nstep:", self.current_step)

                        # pick action from the DQN, epsilon greedy
                        chosenActions, allJQValues = sess.run(
                            [self.collector_mainDQN.allJointsBestActions, self.collector_mainDQN.allJointsQvalues3D],
                            feed_dict={self.collector_mainDQN.inState:np.reshape(initialState, (1, self.stateSize))}
                        )

                        chosenActions = np.reshape(np.array(chosenActions), self.nAJoints)
                        allJQValues = np.reshape(np.array(allJQValues), (self.nAJoints, self.nActionsPerJoint))
                        maxQvalues = allJQValues[range(self.nAJoints), chosenActions] # 1 x nJoints

                        if self.subtask_total_steps <= self.replay_start_size and not self.skip_training:
                            chosenActions = np.random.randint(0, self.nActionsPerJoint, self.nAJoints)
                        else:
                            indices = np.random.rand(self.nAJoints) < epsilon
                            chosenActions[indices] = np.random.randint(0, self.nActionsPerJoint, sum(indices))

                        # perform action and get new state and reward
                        newState, r, done = env.step(chosenActions)
                        # print("\newState: ", newState)
                        # print("\nr: ", r)
                        # print("\ndone: ", done)
                        #add experience to buffer
                        end_multiplier = 0 if self.current_step == self.max_steps_per_episode else 1
                        transition = np.array([initialState, chosenActions, r, newState, end_multiplier])
                        transition = np.reshape(transition, [1, 5]) # 1 x 5
                        episodeBuffer.add(transition) # add step to episode buffer

                        # if not skip_training:
                            # if subtask_total_steps > replay_start_size:
                            #     if subtask_total_steps % train_model_steps_period == 0:
                            #         #train model
                            #         batch = dataset.sample(batch_size)
                            #         # states0, actions0, rewards, states1, dones = batch.T
                            #         states0, actions0, rewards, states1, end_multipliers = batch.T
                            #         states0 = np.vstack(states0)
                            #         actions0 = np.vstack(actions0)
                            #         states1 = np.vstack(states1)
                            #         rewards = np.reshape(rewards, (batch_size, 1))
                            #         # dones = np.reshape(dones, (batch_size, 1))
                            #         end_multipliers = np.reshape(end_multipliers, (batch_size, 1))
                            #         allJBestActions = sess.run(mainDQN.allJointsBestActions, feed_dict={mainDQN.inState:states1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a')) #batch_size x 1
                            #         allJQvalues = sess.run(targetDQN.allJointsQvalues3D, feed_dict={targetDQN.inState:states1}) #feed batch of s' and get batch of Q2(a') # batch_size x 3
                            #         #get Q values of best actions
                            #         allJBestActions_one_hot = np.arange(nActionsPerJoint) == allJBestActions[:, :, None]
                            #         allJBestActionsQValues = np.sum(np.multiply(allJBestActions_one_hot, allJQvalues), axis=2) # batch_size x nJoints
                            #         # end_multiplier = -(dones - 1) # batch_size x 1 ,  equals zero if end of succesful episode
                            #         allJQtargets = np.reshape(rewards + y * allJBestActionsQValues * end_multipliers, (batch_size, nAJoints)) #batch_size x nJoints
                            #         #Update the primary network with our target values.
                            #         _ = sess.run(mainDQN.updateModel, feed_dict={mainDQN.inState:states0, mainDQN.Qtargets:allJQtargets, mainDQN.chosenActions:actions0})
                            #         run_ops(targetOps,sess) #Set the target network to be equal to the primary network.
                        
                        maxQvalues2 = np.reshape(maxQvalues, (self.nAJoints, 1))
                        stateToSave = np.reshape(initialState, (self.stateSize, 1))
                        # if not testing_policy:
                        # end of step, save tracked statistics
                        undisc_return += r
                        sum_of_maxQ += maxQvalues2
                        self.subtask_total_steps += 1
                        # else:
                            # save q values for training logs
                        if (self.subtask_total_steps % self.q_plots_period) >= (self.q_plots_period - self.q_plots_num_of_points):
                            self.statesArray = np.concatenate((self.statesArray, stateToSave), axis=1)
                            self.maxQvaluesArray = np.concatenate((self.maxQvaluesArray, maxQvalues2), axis=1)

                        initialState = newState
                        self.current_step += 1

                        # dont stop the ep. when done, so that the robot stays in the target position
                        if done and not task_completed:
                        #     # if testing_policy:
                        #     #     current_test_success_count += 1
                        #     # else:
                            task_completed = True
                            self.success_steps.append(self.current_step)
                            success_count += 1
                        #     break

                    #episode ended

                    # if i % policy_test_period == 0 and not skip_training and not testing_policy:
                    #     # pause training and test current policy for some episodes
                    #     print("\nTesting policy...")
                    #     subt_test_steps.append(subtask_total_steps)
                    #     current_test_success_count = 0
                    #     testing_policy = True
                    #     skip_training = True
                    #     testing_policy_episode = 1
                    #     epsilon_backup = epsilon

                    # if testing_policy_episode == policy_test_episodes:
                    #     # back to training
                    #     print("\nBack to training.")
                    #     testing_policy = False
                    #     skip_training = False
                    #     epsilon = epsilon_backup
                    #     testing_policy_episode = 0

                    #     current_success_rate = current_test_success_count / policy_test_episodes
                    #     subt_test_success_rates.append(current_success_rate)

                    #     # plot Q plots
                    #     Qplots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                    #     os.makedirs(Qplots_dir_path, exist_ok=True)
                    #     saveQvaluesPlot(Qplots_dir_path, statesArray, maxQvaluesArray, nAJoints)
                    #     statesArray = np.array([]).reshape(stateSize, 0)  # reset q values logs
                    #     maxQvaluesArray = np.array([]).reshape(nAJoints, 0)

                    #     if success_rate_for_subtask_completion and len(subt_test_success_rates)>2:
                    #         if subt_test_success_rates[-1] < subt_test_success_rates[-2]:
                    #             no_progress_count += 1
                    #         else:
                    #             no_progress_count = 0

                    #         if no_progress_count == 5:
                    #             print("\nSuccess rate did not improve, moving on to next task.")
                    #             break

                    # add current episode's list of transitions to dataset
                    if not task_completed:
                        self.success_steps.append(self.max_steps_per_episode)

                    self.dataset.add(episodeBuffer.data)

                    # if testing_policy:
                    #     testing_policy_episode += 1
                    # else:
                    # save tracked statistics
                    self.num_steps_per_ep.append(self.current_step)
                    self.undisc_return_per_ep.append(undisc_return)
                    self.subt_cumul_successes.append(success_count)
                    self.epsilon_per_ep.append(epsilon)

                    averageMaxQ = sum_of_maxQ / self.current_step #nJoints x 1
                    print("averageMaxQ for each joint:\n", averageMaxQ.T)
                    self.average_maxQ_per_ep = np.concatenate((self.average_maxQ_per_ep, averageMaxQ), axis=1)

                    #save the model and log training
                    self.is_saving = True
                    saving_start_time = time.time()
                    if self.current_episode % self.model_saving_period == 0:
                        print("Saving model and plots")
                        save_path = saver.save(sess, self.checkpoint_model_file_path, global_step=self.current_episode)
                        # print("\nepisode: {} steps: {} undiscounted return obtained: {} done: {}".format(self.current_episode, j, undisc_return, done))
                        checkpoints_plots_dir_path = os.path.join(self.current_model_dir_path, "checkpoint_results_ep_" + str(self.current_episode))
                        os.makedirs(checkpoints_plots_dir_path, exist_ok=True)
                        self.savePlots(checkpoints_plots_dir_path)

                    if self.current_episode % self.q_plots_period == 0:
                         # plot Q plots
                        print("Saving Q-plots")
                        Qplots_dir_path = os.path.join(self.current_model_dir_path, "q_plots_ep_" + str(self.current_episode))
                        os.makedirs(Qplots_dir_path, exist_ok=True)
                        # self.lastStatesArray = self.statesArray[:, -self.q_plots_num_of_points:-1]
                        # self.lastMaxQvaluesArray = self.maxQvaluesArray[:, -self.q_plots_num_of_points:-1]
                        self.lastStatesArray = self.statesArray
                        self.lastMaxQvaluesArray = self.maxQvaluesArray
                        self.saveQvaluesPlot(Qplots_dir_path)
                        self.statesArray = np.array([]).reshape(self.stateSize, 0)  # reset q values logs
                        self.maxQvaluesArray = np.array([]).reshape(self.nAJoints, 0)

                    self.is_saving = False

                    saving_end_time = time.time()
                    ep_saving_time = saving_end_time - saving_start_time

                    total_saving_time += ep_saving_time
                    self.current_episode += 1

                collector_end_time = time.time()
                if not self.skip_training:
                    self.coord.request_stop()
                    self.coord.join([trainer_thread])
                #training ended, save results
                end_time = time.time()
                print("Training ended")

                save_path = saver.save(sess, self.trained_model_file_path, global_step=self.num_episodes) #save the trained model
                print("Trained model saved in file: %s" % save_path)

        # save total time and steps to txt file
        collecting_time = collector_end_time - collector_start_time - total_saving_time

        total_training_time_in_secs = end_time - start_time #in seconds
        total_training_time_in_hours = total_training_time_in_secs / 3600
        print('\nTotal training time:', total_training_time_in_hours)
        subt_total_eps = self.current_episode
        collector_freq = self.subtask_total_steps / collecting_time
        self.end_stats_dict["COLLECTOR_total_number_of_steps_executed_subtask"] = self.subtask_total_steps
        self.end_stats_dict["COLLECTOR_total_number_of_episodes_executed_subtask"] = subt_total_eps
        self.end_stats_dict["COLLECTOR__experience_collecting_time_in_sec"] = collecting_time
        self.end_stats_dict["COLLECTOR_env_steps_per_sec"] = collector_freq
        self.end_stats_dict["COLLECTOR_total_saving_time_in_sec"] = total_saving_time

        self.end_stats_dict["total_training_time_in_sec"] = total_training_time_in_secs
        self.end_stats_dict["total_training_time_in_hours"] = total_training_time_in_hours
        if self.end_stats_dict["TRAINER_network_updates_per_sec"]:
            self.end_stats_dict["network_updates_per_env_step"] = self.end_stats_dict["TRAINER_network_updates_per_sec"] / collector_freq

        stats_file_path = os.path.join(self.current_model_dir_path, "end_stats.txt")
        with open(stats_file_path, "w") as stats_file:
            json.dump(self.end_stats_dict, stats_file, sort_keys=True, indent=4)

        # save lists of results for later plots
        lists_to_serialize = ['self.undisc_return_per_ep', 'self.num_steps_per_ep', 'self.subt_cumul_successes', 'self.epsilon_per_ep', 'self.success_steps']
        for list_to_serialize in lists_to_serialize:
            list_json_file = os.path.join(self.current_model_dir_path, list_to_serialize + '.json')
            with open(list_json_file, "w") as json_file:
                json.dump(eval(list_to_serialize), json_file)

        self.savePlots(self.trained_model_plots_dir_path)

        # return save_path, subtask_total_steps, subt_cumul_successes, subt_test_success_rates, subt_test_steps, total_training_time_in_hours  # for visualization and curriculum learning
        return save_path, self.subtask_total_steps, self.subt_cumul_successes, total_training_time_in_hours  # for visualization and curriculum learning

        # # Main thread
        # coord = tf.train.Coordinator()

        # trainer_thread = threading.Thread(name='trainer', target=trainer, args=(coord,))
        # collector_thread = threading.Thread(name='collector', target=collector, args=(coord,))

        # sess = tf.Session()


        # # Start the threads and wait for all of them to stop.
        # trainer_thread.start()
        # collector_thread.start()

        #coord.join([trainer_thread, collector_thread])


    def trainer(self, sess):
        trainer_start_time = time.time()
        while self.subtask_total_steps < self.replay_start_size:
            time.sleep(1)

        print("\n[TRAINER] Dataset has minimum size, training starts..")

        training_loop_start_time = time.time()
        total_network_updates = 0
        while not self.coord.should_stop():
        # while (not self.coord.should_stop()) and (self.current_episode <= self.num_episodes):
        # last_step = self.subtask_total_steps
        # current_update_counter = 0
            if not self.is_saving:
                print("\nUpdate_number: ", total_network_updates)
            # while (not self.coord.should_stop()) and (last_step == self.subtask_total_steps):
                # if current_update_counter < self.max_updates_per_env_step:
                batch = self.dataset.sample(self.batch_size)
                # states0, actions0, rewards, states1, dones = batch.T
                states0, actions0, rewards, states1, end_multipliers = batch.T
                states0 = np.vstack(states0)
                actions0 = np.vstack(actions0)
                states1 = np.vstack(states1)
                rewards = np.reshape(rewards, (self.batch_size, 1))
                # dones = np.reshape(dones, (batch_size, 1))
                end_multipliers = np.reshape(end_multipliers, (self.batch_size, 1))

                allJBestActions = sess.run(self.trainer_mainDQN.allJointsBestActions, feed_dict={self.trainer_mainDQN.inState:states1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a')) #batch_size x 1
                allJQvalues = sess.run(self.trainer_targetDQN.allJointsQvalues3D, feed_dict={self.trainer_targetDQN.inState:states1}) #feed batch of s' and get batch of Q2(a') # batch_size x 3

                #get Q values of best actions
                allJBestActions_one_hot = np.arange(self.nActionsPerJoint) == allJBestActions[:, :, None]
                allJBestActionsQValues = np.sum(np.multiply(allJBestActions_one_hot, allJQvalues), axis=2) # batch_size x nJoints
                # end_multiplier = -(dones - 1) # batch_size x 1 ,  equals zero if end of succesful episode

                allJQtargets = np.reshape(rewards + self.y * allJBestActionsQValues * end_multipliers, (self.batch_size, self.nAJoints)) #batch_size x nJoints

                #Update the primary network with our target values.
                # update_dict = 
                if total_network_updates % self.tensorboard_log_period == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([self.merged, self.trainer_mainDQN.updateModel],
                                          feed_dict={self.trainer_mainDQN.inState:states0,self.trainer_mainDQN.Qtargets:allJQtargets,self.trainer_mainDQN.chosenActions:actions0},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    log_last_step = self.subtask_total_steps
                    if log_last_step != self.subtask_total_steps:
                        self.training_writer.add_run_metadata(run_metadata, 'step%d' % self.subtask_total_steps)
                    self.training_writer.add_summary(summary, self.subtask_total_steps)
                else:
                    _ = sess.run(self.trainer_mainDQN.updateModel, feed_dict={self.trainer_mainDQN.inState:states0,
                                                                               self.trainer_mainDQN.Qtargets:allJQtargets,
                                                                               self.trainer_mainDQN.chosenActions:actions0})
                self.softUpdateTarget(sess) #Soft update of the target network towards the main network.
                # current_update_counter +=1
                total_network_updates +=1
                #else: wait for the next environment step

        training_loop_end_time = time.time()
        print("\n[TRAINER] Thread ended.")
        waiting_time = training_loop_start_time - trainer_start_time
        training_time = training_loop_end_time - training_loop_start_time
        trainer_freq = total_network_updates / training_time
        self.end_stats_dict["TRAINER_total_network_updates"] = total_network_updates
        self.end_stats_dict["TRAINER_waiting_time_in_sec"] = waiting_time
        self.end_stats_dict["TRAINER_training_time_after_min_dataset_in_sec"] = training_time
        self.end_stats_dict["TRAINER_network_updates_per_sec"] = trainer_freq


    def saveRewardFunction(self, robotenv, dir_path):
        fig = plt.figure()
        distance = np.arange(0., 3., 0.05)
        rewards = robotenv.distance2reward(distance)
        plt.plot(distance, rewards, linewidth=0.5)
        plt.ylabel('reward')
        plt.xlabel('distance to goal (m)')
        plt.title('Reward function')
        plot_file = os.path.join(dir_path, 'reward_function.svg')
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close()


    def savePlot(self, dir_path, var_value_per_ep, ylabel_str, title_str, name):
        fig = plt.figure()
        episodes = range(1, len(var_value_per_ep) + 1)  # start at episode 1
        plt.plot(episodes, var_value_per_ep, linewidth=0.5)
        plt.xlabel('episode')
        plt.ylabel(ylabel_str)
        plt.title(title_str)
        plot_file = os.path.join(dir_path, name)
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close()


    def saveQvaluesPlot(self, dir_path):
        # statesArray = maxQvaluesArray = nJoints x ?
        for i in range(self.nAJoints):  # first state elements are the joint angles
            fig = plt.figure()
            plt.scatter(self.lastStatesArray[i], self.lastMaxQvaluesArray[i]) # (normalized_angles, maxQvalues)
            plt.xlabel('normalized angle')
            plt.ylabel('max Q-value')
            plt.title('Max Q-values for angles observed, joint' + str(i + 1))
            plot_file = os.path.join(dir_path, 'angles_Q_values_joint' + str(i + 1) + '.svg')
            fig.savefig(plot_file, bbox_inches='tight')
            plt.close()


    def savePlots(self, dir_path):
        #note: "per_ep" in variable names were omitted
        # discounted returns for each episode
        self.savePlot(dir_path, self.undisc_return_per_ep, "undisc. return", "Undiscounted return obtained", "undisc_returns.svg")

        # steps performed in each episode
        self.savePlot(dir_path, self.num_steps_per_ep, "steps", "Steps performed per episode", "steps.svg")

        # number of success so far, for each episode
        self.savePlot(dir_path, self.subt_cumul_successes, "successes", "Number of successes", "successes.svg")
        
        # epsilon evolution
        self.savePlot(dir_path, self.epsilon_per_ep, "epsilon value", "Epsilon updates", "epsilons.svg")

        # success steps
        self.savePlot(dir_path, self.success_steps, "success step", "Step of first success", "success_steps.svg")

        # average (over steps, for each episode) of maxQ
        for i in range(0, self.average_maxQ_per_ep.shape[0]):
            self.savePlot(dir_path, self.average_maxQ_per_ep[i], "average maxQ", "Average maxQ per episode, joint" + str(i + 1), "average_q" + str(i + 1) + ".svg")

    # def collector(self, sess):
