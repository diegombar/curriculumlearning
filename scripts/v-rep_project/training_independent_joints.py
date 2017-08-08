# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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

# Load environment
from robotenv import RobotEnv

h_params = {} # params to save in txt file:

def saveRewardFunction(robotenv, dir_path):
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

def savePlot(dir_path, var_value_per_ep, ylabel_str, title_str, name):
    fig = plt.figure()
    episodes = range(1, len(var_value_per_ep) + 1) # start at episode 1
    plt.plot(var_value_per_ep, linewidth=0.5)
    plt.xlabel('episode')
    plt.ylabel(ylabel_str)
    plt.title(title_str)
    plot_file = os.path.join(dir_path, name)
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()

def saveQvaluesPlot(dir_path, statesArray, maxQvaluesArray, nJoints=6):
    # statesArray = maxQvaluesArray = nJoints x ?
    for i in range(nJoints): #first state elements have to be the joint angles
        normalized_angles = statesArray[i]
        maxQvalues = maxQvaluesArray[i]
        fig = plt.figure()
        plt.scatter(normalized_angles, maxQvalues)
        plt.xlabel('normalized angle')
        plt.ylabel('max Q-value')
        plt.title('Max Q-values for angles observed, joint'+ str(i + 1))
        plot_file = os.path.join(dir_path, 'angles_Q_values_joint' + str(i + 1) + '.svg')
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close()

def savePlots(
  dir_path, undisc_returns, num_steps, 
  successes, epsilons, avg_maxQs):
    #note: "per_ep" in variable names were omitted
    # discounted returns for each episode
    savePlot(dir_path, undisc_returns, "undisc. return", "Undiscounted return obtained", "undisc_returns.svg")

    # steps performed in each episode
    savePlot(dir_path, num_steps, "steps", "Steps performed per episode", "steps.svg")

    # number of success so far, for each episode
    savePlot(dir_path, successes, "successes", "Number of successes", "successes.svg")
    
    # epsilon evolution
    savePlot(dir_path, epsilons, "epsilon value", "Epsilon updates", "epsilons.svg")

    # average (over steps, for each episode) of maxQ
    avg_maxQ1s, avg_maxQ2s, avg_maxQ3s, avg_maxQ4s, avg_maxQ5s, avg_maxQ6s = avg_maxQs
    savePlot(dir_path, avg_maxQ1s, "average maxQ", "Average maxQ per episode, joint 1", "average_q1.svg")
    savePlot(dir_path, avg_maxQ2s, "average maxQ", "Average maxQ per episode, joint 2", "average_q2.svg")
    savePlot(dir_path, avg_maxQ3s, "average maxQ", "Average maxQ per episode, joint 3", "average_q3.svg")
    savePlot(dir_path, avg_maxQ4s, "average maxQ", "Average maxQ per episode, joint 4", "average_q4.svg")
    savePlot(dir_path, avg_maxQ5s, "average maxQ", "Average maxQ per episode, joint 5", "average_q5.svg")
    savePlot(dir_path, avg_maxQ6s, "average maxQ", "Average maxQ per episode, joint 6", "average_q6.svg")

# experience replay dataset, experience = (s,a,r,s',done)
class experience_dataset():
    def __init__(self, size):
        self.data = []
        self.size = size

    #add experience = list of transitions, freeing space if needed
    def add(self, experience):
        excess = len(self.data) + len(experience) - self.size
        if excess > 0:
            self.data[0:excess] = []
        self.data.extend(experience) 

    # randomly sample an array of transitions (s,a,r,s',done)
    def sample(self,sample_size):
        sample = np.array(random.sample(self.data,sample_size))
        return np.reshape(sample, [sample_size,5])

class DQN():
    def __init__(self, nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate, use_variable_names=True, previous_norm=False):
        self.nJoints = 6
        self.nActionsPerJoint = nActions // self.nJoints
        self.inState = tf.placeholder(shape=[None,stateSize], dtype=tf.float32, name='state') #batch_size x stateSize

        self.variable_dict = {}

        nHidden = num_neurons_per_hidden
        print
        if use_variable_names:
            # list of layer sizes
            neuronsPerLayer = [num_neurons_per_hidden] * (num_hidden_layers + 2)
            neuronsPerLayer[0] = stateSize
            neuronsPerLayer[-1] = nActions

            # initialize params
            self.weights = []
            self.biases = []
            self.hidden_layers = []
            for i in range(len(neuronsPerLayer) - 1):
                with tf.name_scope('layer'+str(i)) as scope:
                    weight_name = "weight" + str(i)
                    bias_name = "bias" + str(i)
                    w = tf.Variable(tf.truncated_normal([neuronsPerLayer[i], neuronsPerLayer[i+1]], mean=0.0, stddev=0.1), name=weight_name)
                    b = tf.Variable(tf.constant(0.1, shape=[1]), name=bias_name)
                    self.weights.append(w)
                    self.biases.append(b)
                    self.variable_dict[weight_name] = self.weights[-1]
                    self.variable_dict[bias_name] = self.biases[-1]
                    if i == 0:
                        if previous_norm:
                            h_layer1 = tf.nn.relu(tf.matmul(2 * self.inState, self.weights[0]) + self.biases[0], name="neuron_activations" + str(i))
                        else:
                            h_layer1 = tf.nn.relu(tf.matmul(self.inState, self.weights[0]) + self.biases[0], name="neuron_activations" + str(i))
                        self.hidden_layers.append(h_layer1)
                    elif i<(len(neuronsPerLayer) - 2):
                        h_layer = tf.nn.relu(tf.matmul(self.hidden_layers[-1], self.weights[-1]) + self.biases[-1], name="neuron_activations" + str(i))
                        self.hidden_layers.append(h_layer)
                    else:
                        self.allJointsQvalues = tf.add(tf.matmul(self.hidden_layers[-1], self.weights[-1]),self.biases[-1], name="q_values") # Q values for all actions given inState, #batch_size x nActions
            # self.variable_dict = {
            #                 "weight0":self.weights[0],
            #                 "weight1":self.weights[1],
            #                 "weight1":self.weights[2],
            #                 "bias0":self.biases[0],
            #                 "bias1":self.biases[1],
            #                 "bias2":self.biases[2],
            #             }
        else:
            self.W0 = tf.Variable(tf.truncated_normal([stateSize, nHidden], mean=0.0, stddev=0.1),
                # name="weight0"
                )
            self.W1 = tf.Variable(tf.truncated_normal([nHidden, nHidden], mean=0.0, stddev=0.1),
                # name="weight1"
                )
            self.W2 = tf.Variable(tf.truncated_normal([nHidden, nActions], mean=0.0, stddev=0.1),
                # name="weight2"
                )
            self.b0 = tf.Variable(tf.constant(0.1, shape=[1]),
                # name="bias0"
                )
            self.b1 = tf.Variable(tf.constant(0.1, shape=[1]),
                # name="bias1"
                )
            self.b2 = tf.Variable(tf.constant(0.1, shape=[1]),
                # name="bias2"
                )

            self.variable_dict = {
                                     "Variable":self.W0,
                                     "Variable_1":self.W1,
                                     "Variable_2":self.W2,
                                     "Variable_3":self.b0,
                                     "Variable_4":self.b1,
                                     "Variable_5":self.b2,
                                 }

            # layers
            if previous_norm:
                self.hidden1 = tf.nn.relu(tf.matmul(2 * self.inState, self.W0) + self.b0)
            else:
                self.hidden1 = tf.nn.relu(tf.matmul(self.inState, self.W0) + self.b0)
            self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W1) + self.b1)
            self.allJointsQvalues = tf.matmul(self.hidden2, self.W2) + self.b2 # Q values for all actions given inState, #batch_size x nActions

        # self.j1Qvalues, self.j2Qvalues, self.j3Qvalues, self.j4Qvalues, self.j5Qvalues, self.j6Qvalues = tf.split(self.allJointsQvalues, self.nJoints, axis=1) # batch_size x (nJoints x actionsPerJoint)

        # get actions of highest Q value for each joint
        self.allJointsQvalues3D = tf.reshape(self.allJointsQvalues, [-1, self.nJoints, self.nActionsPerJoint]) # batch_size x nJoints x actionsPerJoint
        self.allJointsBestActions = tf.argmax(self.allJointsQvalues3D, axis=2, name='best_actions') # batch_size x nJoints

        #get Q target values
        self.Qtargets = tf.placeholder(shape=[None, self.nJoints], dtype=tf.float32, name='q_targets') #batch_size x nJoints

        # get batch of executed actions a0
        self.chosenActions = tf.placeholder(shape=[None, self.nJoints],dtype=tf.int32, name='chosen_actions') #batch_size x nJoints

        #get Q values corresponding to executed actions (i.e. Q values to update)
        with tf.name_scope('q_values_to_update') as scope:
            self.chosenAs_onehot = tf.one_hot(self.chosenActions, self.nActionsPerJoint, dtype=tf.float32) #batch_size x nJoints x nActionsPerJoint
            self.chosenActionsQvalues = tf.reduce_sum(tf.multiply(self.allJointsQvalues3D, self.chosenAs_onehot), axis=2, name='executed_actions_q_values') #element-wise multiplication

        # loss by taking the sum of squares difference between the target and predicted Q values
        self.error = tf.square(self.Qtargets - self.chosenActionsQvalues, name='error') #element-wise
        self.loss = tf.reduce_mean(self.error, name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
        self.updateModel = self.optimizer.minimize(self.loss)

# update target DQN weights
def updateTargetGraph(tfTrainables,tau):
    total_vars = len(tfTrainables)
    op_holder = []
    #first half of tfTrainables corresponds to mainQN, second half to targetDQN
    for idx,mainNetVar in enumerate(tfTrainables[0:total_vars // 2]):
        targetNetVar = tfTrainables[idx + total_vars // 2]
        op = targetNetVar.assign(tau * mainNetVar.value() + (1 - tau)*targetNetVar.value())
        op_holder.append(op)
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def trainDQL(
  experiment_folder_name,
  num_hidden_layers, num_neurons_per_hidden,
  num_episodes, max_steps_per_episode, e_min,
  task,
  model_saving_period=100,
  lrate=1E-6,
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
  policy_test_period=100, # episodes
  test_success_rate_list=None, # policy success rate list
  test_step_numbers=None, # track number of steps before each test
  success_rate_for_subtask_completion=False
  ):

    # hyper params to save to txt file
    h_params["experiment_folder_name"] = experiment_folder_name
    h_params["showGUI"] = showGUI
    h_params['num_hidden_layers'] = num_hidden_layers # not counting output layer
    h_params['neurons_per_hidden_layer'] = num_neurons_per_hidden  #mnih: 512 for dense hidden layer
    h_params['num_episodes'] = num_episodes
    h_params['max_steps_per_episode'] = max_steps_per_episode
    h_params['model_saving_period'] = model_saving_period
    h_params['q_values_log_period'] = q_values_log_period = max(model_saving_period // 5, 1)
    h_params['e_min'] = e_min
    h_params['task'] = task
    h_params['batch_size'] = batch_size #mnih=32
    h_params['replay_start_size'] = replay_start_size # steps to fill dataset with random actions mnih=5E4
    h_params['replay_memory_size'] = replay_memory_size # in steps #mnih: 1E6
    h_params["joint_velocity"] = velocity
    h_params['use_variable_names'] = use_variable_names #use non-default names for variables

    h_params['hostname'] = socket.gethostname()

    # load model if path is specified
    load_model = False
    if model_to_load_file_path is not None:
        # e.g.
        # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","saved_checkpoints","checkpoint_model-400")
        # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","trained_model","final_model-2000")
        h_params["model_to_load_file_path"] = model_to_load_file_path
        load_model = True

    h_params['load_model'] = load_model
    h_params['skip_training'] = skip_training #skip for visualization, do not skip for curriculum learning/pre-training

    if notes is not None: h_params['notes'] = notes


    if replay_start_size <= max_steps_per_episode or replay_start_size < batch_size:
        print("WARNING: replay_start_size must be greater than max_steps_per_episode and batch_size")

    # create folders to save results
    current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
    all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
    experiment_dir_path = os.path.join(all_models_dir_path, experiment_folder_name)
    timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
    current_model_dir_path = os.path.join(experiment_dir_path, "model_and_results_" + timestr)
    trained_model_plots_dir_path = os.path.join(current_model_dir_path, "trained_model_results")
    checkpoints_dir_path = os.path.join(current_model_dir_path, "saved_checkpoints")
    trained_model_dir_path = os.path.join(current_model_dir_path, "trained_model")
    checkpoint_model_file_path = os.path.join(checkpoints_dir_path, "checkpoint_model")
    trained_model_file_path = os.path.join(trained_model_dir_path, "final_model")
    log_file_path = os.path.join(current_model_dir_path,"logs")
    
    for new_directory in [trained_model_plots_dir_path, checkpoints_dir_path, trained_model_dir_path, log_file_path]:
        os.makedirs(new_directory, exist_ok=True)

    # save git commit hash
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    h_params["_commit_hash"] = git_hash.decode("utf-8").strip()

    #plot and save reward function
    h_params['rewards_normalizer'] = rewards_normalizer = 0.1
    distanceOfRewardCloseToZero = 1.0
    h_params['rewards_decay_rate'] = rewards_decay_rate = 1.0/ (distanceOfRewardCloseToZero / 5) #=1/0.33 i.e. near 0 at 5 * 0.33 = 1.65m away

    # recursive exponential decay for epsilon
    h_params['e_max'] = e_max = 1.0 #P(random action in at least one joint) = 1- (1 - epsilon)**nJoints
    h_params['e_tau'] = e_tau = (num_episodes - (replay_start_size // max_steps_per_episode)) / 5 # time constant in episodes, close to final value at 5 e_tau
    addEFactor = 1.0 - (1.0 / e_tau)

    h_params['train_model_steps_period'] = train_model_steps_period = 4 # mnih = 4, period of mini-batch sampling and training
    h_params['update_target_net_rate_tau'] = tau = 0.001 # rate to update target network toward main network
    h_params['learning_rate'] = lrate #= 1E-6
    h_params['discount_factor'] = y = 0.99 # mnih:0.99
    h_params['policy_test_period'] = policy_test_period #episodes
    h_params['policy_test_num_of_test_episodes '] = policy_test_episodes = 10 # episodes


    with RobotEnv(task=task,
        targetPosition=targetRelativePos,
        rewards_normalizer=rewards_normalizer,
        rewards_decay_rate=rewards_decay_rate,
        showGUI=showGUI,
        velocity=velocity) as env:
        tf.reset_default_graph()
        nActionsPerJoint = 3
        h_params['state_size'] = stateSize = env.observation_space_size
        h_params['number_of_actions'] = nActions = env.action_space_size
        h_params['number_of_joints'] = nJoints = nActions // nActionsPerJoint

        saveRewardFunction(env, current_model_dir_path)

        mainDQN = DQN(nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate, use_variable_names, previous_norm)
        targetDQN = DQN(nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate, use_variable_names, previous_norm)

        # save txt file with hyper parameters
        h_params_file_path = os.path.join(current_model_dir_path, "hyper_params.txt")
        with open(h_params_file_path, "w") as h_params_file:
            json.dump(h_params, h_params_file, sort_keys=True, indent=4)

        # initialize and create variables saver
        init = tf.global_variables_initializer()

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

        saver = tf.train.Saver(mainDQN.variable_dict)
        trainables = tf.trainable_variables()
        targetOps = updateTargetGraph(trainables,tau)

        with tf.Session() as sess:
            print("Starting training...")
            start_time = time.time()
            # summary data logs for TensorBoard
            training_writer = tf.summary.FileWriter(log_file_path + '/training', sess.graph)
            sess.run(init)

            if load_model:
                print('\n################ LOADING SAVED MODEL ################')
                saver.restore(sess, model_to_load_file_path)
                print('\n######## SAVED MODEL WAS SUCCESSFULLY LOADED ########')

            updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

            #initialize epsilon
            epsilon = e_min
            if not skip_training:
                addE = e_max - e_min
                epsilon = e_min + addE
                dataset = experience_dataset(replay_memory_size)

            total_steps = 0
            num_steps_per_ep = []
            undisc_return_per_ep = []
            success_count = 0
            successes = []
            epsilon_per_ep = []
            average_maxQ_per_ep = np.array([]).reshape(nJoints,0)
            statesArray = np.array([]).reshape(stateSize,0)
            maxQvaluesArray = np.array([]).reshape(nJoints,0)

            # success rate initialization:
            if test_success_rate_list is None:
                test_success_rate_list = [0]
            if test_step_numbers is None:
                test_step_numbers = [0]
            testing_policy_episode = 0
            testing_policy = False
            epsilon_backup = 0
            no_progress_count = 0

            i = 1
            while i <= num_episodes:
                print("\nepisode number ", i)

                if skip_training:
                    epsilon = e_min
                elif total_steps > replay_start_size:
                    # decay epsilon
                    addE *= addEFactor
                    epsilon = e_min + addE

                initialState = env.reset() # reset environment and get first observation
                undisc_return = 0
                sum_of_maxQ = np.zeros((nJoints,1))
                done = False
                if not skip_training: episodeBuffer = experience_dataset(replay_memory_size) # temporary buffer
                
                j = 1
                while j <= max_steps_per_episode:
                    print("\nstep:", j)

                    # pick action from the DQN, epsilon greedy
                    chosenActions, allJQValues = sess.run(
                        [mainDQN.allJointsBestActions, mainDQN.allJointsQvalues3D],
                        feed_dict={mainDQN.inState:np.reshape(initialState, (1, stateSize))}
                    )

                    chosenActions = np.reshape(np.array(chosenActions), nJoints)
                    allJQValues = np.reshape(np.array(allJQValues), (nJoints, nActionsPerJoint))
                    maxQvalues = allJQValues[range(nJoints), chosenActions] # 1 x nJoints

                    if total_steps <= replay_start_size and not skip_training:
                        chosenActions = np.random.randint(0, nActionsPerJoint, nJoints)
                    else:
                        indices = np.random.rand(6) < epsilon
                        chosenActions[indices] = np.random.randint(0, nActionsPerJoint, sum(indices))

                    # perform action and get new state and reward
                    newState, r, done = env.step(chosenActions)

                    if not skip_training:
                        #add experience to buffer
                        transition = np.array([initialState, chosenActions, r, newState, done])
                        transition = np.reshape(transition, [1, 5]) # 1 x 5
                        episodeBuffer.add(transition) # add step to episode buffer

                        if total_steps > replay_start_size:
                            if total_steps % train_model_steps_period == 0:
                                #train model
                                batch = dataset.sample(batch_size)
                                states0, actions0, rewards, states1, dones = batch.T
                                states0 = np.vstack(states0)
                                actions0 = np.vstack(actions0)
                                states1 = np.vstack(states1)
                                rewards = np.reshape(rewards, (batch_size, 1))
                                dones = np.reshape(dones, (batch_size, 1))

                                allJBestActions = sess.run(mainDQN.allJointsBestActions, feed_dict={mainDQN.inState:states1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a')) #batch_size x 1
                                allJQvalues = sess.run(targetDQN.allJointsQvalues3D, feed_dict={targetDQN.inState:states1}) #feed btach of s' and get batch of Q2(a') # batch_size x 3

                                #get Q values of best actions
                                allJBestActions_one_hot = np.arange(nActionsPerJoint) == allJBestActions[:,:,None]
                                allJBestActionsQValues = np.sum(np.multiply(allJBestActions_one_hot, allJQvalues), axis=2) # batch_size x nJoints
                                end_multiplier = -(dones - 1) # batch_size x 1

                                allJQtargets = np.reshape(rewards + y * allJBestActionsQValues * end_multiplier, (batch_size,nJoints)) #batch_size x nJoints

                                #Update the primary network with our target values.
                                _ = sess.run(mainDQN.updateModel, feed_dict={mainDQN.inState:states0, mainDQN.Qtargets:allJQtargets, mainDQN.chosenActions:actions0})
                                updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

                    if not testing_policy:
                        # end of step, save tracked statistics
                        undisc_return += r
                        maxQvalues2 = np.reshape(maxQvalues, (nJoints, 1))
                        stateToSave = np.reshape(initialState, (stateSize, 1))
                        sum_of_maxQ += maxQvalues2
                        total_steps += 1
                    else:
                        # save q values for training logs
                        # if total_steps % q_values_log_period == 0:
                        statesArray = np.concatenate((statesArray, stateToSave), axis=1)
                        maxQvaluesArray = np.concatenate((maxQvaluesArray, maxQvalues2), axis=1)

                    initialState = newState
                    j += 1

                    if done:
                        if testing_policy:
                            current_test_success_count += 1
                        else:
                            success_count += 1
                        break

                #episode ended

                if testing_policy_episode == policy_test_episodes:
                    # back to training
                    print("\nBack to training.")
                    testing_policy = False
                    skip_training = False
                    epsilon = epsilon_backup
                    testing_policy_episode = 0

                    current_success_rate = current_test_success_count / policy_test_episodes
                    test_success_rate_list.append(current_success_rate)

                    # plot Q plots
                    Qplots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                    os.makedirs(Qplots_dir_path, exist_ok=True)
                    saveQvaluesPlot(Qplots_dir_path, statesArray, maxQvaluesArray)
                    statesArray = np.array([]).reshape(stateSize,0) # reset q values logs
                    maxQvaluesArray = np.array([]).reshape(nJoints,0)

                    if success_rate_for_subtask_completion:
                        if test_success_rate_list[-1] < (test_success_rate_list[-2] * 1.1):
                            no_progress_count += 1
                        else:
                            no_progress_count = 0

                        if no_progress_count == 3:
                            print("\nSuccess rate did not improve, moving on to next task.")
                            break

                # add current episode's list of transitions to dataset
                if not skip_training: 
                    dataset.add(episodeBuffer.data)

                if testing_policy:
                    testing_policy_episode += 1
                else:
                    # save tracked statistics
                    num_steps_per_ep.append(j)
                    undisc_return_per_ep.append(undisc_return)
                    successes.append(success_count)
                    epsilon_per_ep.append(epsilon)

                    averageMaxQ = sum_of_maxQ / j #nJoints x 1
                    print("averageMaxQ for each joint:\n", averageMaxQ.T)
                    average_maxQ_per_ep = np.concatenate((average_maxQ_per_ep, averageMaxQ), axis=1)

                    #save the model and log training
                    if i % model_saving_period == 0:
                        print("Saving model and results")
                        save_path = saver.save(sess, checkpoint_model_file_path, global_step=i)
                        print("\nepisode: {} steps: {} undiscounted return obtained: {} done: {}".format(i, j, undisc_return, done))
                        checkpoints_plots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                        os.makedirs(checkpoints_plots_dir_path, exist_ok=True)
                        savePlots(checkpoints_plots_dir_path, undisc_return_per_ep, num_steps_per_ep,
                                  successes, epsilon_per_ep, average_maxQ_per_ep)

                    if i % policy_test_period == 0 and not skip_training and not testing_policy:
                        # pause training and test current policy for some episodes
                        print("\nTesting policy...")
                        test_step_numbers.append(total_steps)
                        current_test_success_count = 0
                        testing_policy = True
                        skip_training = True
                        testing_policy_episode = 1
                        epsilon_backup = epsilon

                    i += 1

            #training ended, save results
            end_time = time.time()
            print("Training ended")

            save_path = saver.save(sess, trained_model_file_path, global_step=num_episodes) #save the trained model
            print("Trained model saved in file: %s" % save_path)

    # save total time and steps to txt file
    total_training_time = end_time - start_time #in seconds
    print('\nTotal training time:', total_training_time)
    end_stats_dict = {"total_number_of_steps_executed":total_steps}
    end_stats_dict = {"total_number_of_episodes_executed":i}
    end_stats_dict["total_training_time_in_secs"] = total_training_time
    stats_file_path = os.path.join(current_model_dir_path, "end_stats.txt")
    with open(stats_file_path, "w") as stats_file:
        json.dump(end_stats_dict, stats_file, sort_keys=True, indent=4)

    # save lists of results for later plots
    lists_to_serialize = ['undisc_return_per_ep', 'num_steps_per_ep', 'successes', 'epsilon_per_ep']
    for list_to_serialize in lists_to_serialize:
        list_json_file = os.path.join(current_model_dir_path, list_to_serialize + '.json')
        with open(list_json_file, "w") as json_file:
            json.dump(eval(list_to_serialize), json_file)

    savePlots(trained_model_plots_dir_path, undisc_return_per_ep, num_steps_per_ep, successes, epsilon_per_ep, average_maxQ_per_ep)

    return save_path, test_success_rate_list, test_step_numbers  # for visualization and curriculum learning
