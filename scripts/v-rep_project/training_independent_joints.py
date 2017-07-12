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

# Load environment
from robotenv import RobotEnv

h_params = {} # params to save in txt file:

def saveRewardFunction(normalizer, decay_rate, dir_path):
    fig = plt.figure()
    d = np.arange(0., 3., 0.05)
    rewards = normalizer * np.exp(-decay_rate * d)
    plt.plot(d, rewards, linewidth=0.5)
    plt.ylabel('reward')
    plt.xlabel('distance to goal (m)')
    plt.title('Reward function')
    plot_file = os.path.join(dir_path, 'reward_function.svg')
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()

def savePlot(var_value_per_ep, ylabel_str, title_str, dir_path, name):
    fig = plt.figure()
    plt.plot(var_value_per_ep, linewidth=0.5)
    plt.ylabel(ylabel_str)
    plt.xlabel('episode')
    plt.title(title_str)
    plot_file = os.path.join(dir_path, name)
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()

def savePlots(
  dir_path, undisc_returns, num_steps, 
  successes, epsilons, avg_maxQs):
    #note: "per_ep" in variable names were omitted
    # discounted returns for each episode
    savePlot(undisc_returns, "undisc. return", "Undiscounted return obtained", dir_path, "undisc_returns.svg")

    # steps performed in each episode
    savePlot(num_steps, "steps", "Steps performed per episode", dir_path, "steps.svg")

    # number of success so far, for each episode
    savePlot(successes, "successes", "Number of successes", dir_path, "successes.svg")
    
    # epsilon evolution
    savePlot(epsilons, "epsilon value", "Epsilon updates", dir_path, "epsilons.svg")

    # average (over steps, for each episode) of maxQ
    avg_maxQ1s, avg_maxQ2s, avg_maxQ3s, avg_maxQ4s, avg_maxQ5s, avg_maxQ6s = avg_maxQs
    savePlot(avg_maxQ1s, "average maxQ", "Average maxQ per episode, joint 1", dir_path, "average_q1.svg")
    savePlot(avg_maxQ2s, "average maxQ", "Average maxQ per episode, joint 2", dir_path, "average_q2.svg")
    savePlot(avg_maxQ3s, "average maxQ", "Average maxQ per episode, joint 3", dir_path, "average_q3.svg")
    savePlot(avg_maxQ4s, "average maxQ", "Average maxQ per episode, joint 4", dir_path, "average_q4.svg")
    savePlot(avg_maxQ5s, "average maxQ", "Average maxQ per episode, joint 5", dir_path, "average_q5.svg")
    savePlot(avg_maxQ6s, "average maxQ", "Average maxQ per episode, joint 6", dir_path, "average_q6.svg")

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
    def __init__(self, nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate):
        self.nJoints = 6
        self.nActionsPerJoint = nActions // self.nJoints

        self.inState = tf.placeholder(shape=[None,stateSize], dtype=tf.float32) #batch_size x stateSize

        # list of layer sizes
        neuronsPerLayer = [num_neurons_per_hidden] * (num_hidden_layers + 2)
        neuronsPerLayer[0] = stateSize
        neuronsPerLayer[-1] = nActions

        # initialize params
        self.weights = []
        self.biases = []
        self.hidden_layers = []
        for i in range(len(neuronsPerLayer) - 1):
            w = tf.Variable(tf.truncated_normal([neuronsPerLayer[i], neuronsPerLayer[i+1]], mean=0.0, stddev=0.1), name="weight" + str(i))
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="bias" + str(i))
            self.weights.append(w)
            self.biases.append(b)
            if i == 0:
                self.hidden_layers.append(tf.nn.relu(tf.matmul(self.inState, self.weights[0]) + self.biases[0]))
            elif i<(len(neuronsPerLayer) - 2):
                self.hidden_layers.append(tf.nn.relu(tf.matmul(self.hidden_layers[-1], self.weights[-1]) + self.biases[-1]))
            else:
                self.allJointsQvalues = tf.matmul(self.hidden_layers[-1], self.weights[-1]) + self.biases[-1] # Q values for all actions given inState, #batch_size x nActions

        # self.W0 = tf.Variable(tf.truncated_normal([stateSize, nHidden], mean=0.0, stddev=0.1), name="weights0")
        # self.W1 = tf.Variable(tf.truncated_normal([nHidden, nHidden], mean=0.0, stddev=0.1), name="weights1")
        # self.W2 = tf.Variable(tf.truncated_normal([nHidden, nActions], mean=0.0, stddev=0.1), name="weights2")

        # self.b0 = tf.Variable(tf.constant(0.1, shape=[1]), name="bias0")
        # self.b1 = tf.Variable(tf.constant(0.1, shape=[1]), name="bias1")
        # self.b2 = tf.Variable(tf.constant(0.1, shape=[1]), name="bias2")

        # layers
        # self.hidden1 = tf.nn.relu(tf.matmul(self.inState, self.W0) + self.b0)
        # self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W1) + self.b1)
        # self.allJointsQvalues = tf.matmul(self.hidden2, self.W2) + self.b2 # Q values for all actions given inState, #batch_size x nActions

        # self.j1Qvalues, self.j2Qvalues, self.j3Qvalues, self.j4Qvalues, self.j5Qvalues, self.j6Qvalues = tf.split(self.allJointsQvalues, self.nJoints, axis=1) # batch_size x (nJoints x actionsPerJoint)
        #get each one batch_size x actionsPerJoint

        self.allJointsQvalues3D = tf.reshape(self.allJointsQvalues, [-1, self.nJoints, self.nActionsPerJoint]) # batch_size x nJoints x actionsPerJoint
        self.allJointsBestActions = tf.argmax(self.allJointsQvalues3D, axis=2) # batch_size x nJoints

        # get batch of chosen actions a0
        self.chosenActions = tf.placeholder(shape=[None, self.nJoints],dtype=tf.int32) #batch_size x nJoints
        #select Q values for chosen actions a0
        self.chosenAs_onehot = tf.one_hot(self.chosenActions, self.nActionsPerJoint, dtype=tf.float32) #batch_size x nJoints x nActionsPerJoint
        self.chosenActionsQvalues = tf.reduce_sum(tf.multiply(self.allJointsQvalues3D, self.chosenAs_onehot), axis=2) #element-wise multiplication

        # action with highest Q given inState

        # loss by taking the sum of squares difference between the target and prediction Q values.
        self.Qtargets = tf.placeholder(shape=[None, self.nJoints], dtype=tf.float32) #batch_size x nJoints
        self.error = tf.square(self.Qtargets - self.chosenActionsQvalues) #element-wise
        self.loss = tf.reduce_mean(self.error)
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
  num_hidden_layers, num_neurons_per_hidden,
  num_episodes, max_steps_per_episode, e_min, model_saving_period,
  batch_size, replay_start_size, replay_memory_size,
  showGUI=True,
  velocity=0.3,
  model_to_load_file_path=None,
  notes=None):

    # hyper params to save to txt file
    h_params["showGUI"] = showGUI
    h_params['num_hidden_layers'] = num_hidden_layers # not counting output layer
    h_params['neurons_per_hidden_layer'] = num_neurons_per_hidden  #mnih: 512 for dense hidden layer
    h_params['num_episodes'] = num_episodes
    h_params['max_steps_per_episode'] = max_steps_per_episode
    h_params['e_min'] = e_min
    h_params['batch_size'] = batch_size #mnih=32
    h_params['replay_start_size'] = replay_start_size # steps to fill dataset with random actions mnih=5E4
    h_params['replay_memory_size'] = replay_memory_size # in steps #mnih: 1E6
    h_params["joint_velocity"] = velocity

    if notes is not None: h_params['notes'] = notes

    if replay_start_size <= max_steps_per_episode or replay_start_size < batch_size:
        print("WARNING: replay_start_size must be greater than max_steps_per_episode and batch_size")

    # create folders to save results
    current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
    all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
    timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
    current_model_dir_path = os.path.join(all_models_dir_path, "model_and_results_" + timestr)
    trained_model_plots_dir_path = os.path.join(current_model_dir_path, "trained_model_results")
    checkpoints_dir_path = os.path.join(current_model_dir_path, "saved_checkpoints")
    trained_model_dir_path = os.path.join(current_model_dir_path, "trained_model")
    checkpoint_model_file_path = os.path.join(checkpoints_dir_path, "checkpoint_model")
    trained_model_file_path = os.path.join(trained_model_dir_path, "final_model")
    
    for new_directory in [trained_model_plots_dir_path, checkpoints_dir_path, trained_model_dir_path]:
        os.makedirs(new_directory, exist_ok=True)

    # save git commit hash
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    h_params["_commit_hash"] = git_hash.decode("utf-8").strip()

    #plot and save reward function
    h_params['rewards_normalizer'] = rewards_normalizer = 0.1
    distanceOfRewardCloseToZero = 1.0
    h_params['rewards_decay_rate'] = rewards_decay_rate = 1.0/ (distanceOfRewardCloseToZero / 5) #=1/0.33 i.e. near 0 at 5 * 0.33 = 1.65m away

    saveRewardFunction(rewards_normalizer, rewards_decay_rate, current_model_dir_path)

    # load model if path is specified
    load_model = False
    skip_training = False
    if model_to_load_file_path is not None:
        # e.g.
        # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","saved_checkpoints","checkpoint_model-400")
        # model_to_load_file_path = os.path.join(all_models_dir_path,"model_and_results_2017-Jul-07_20-22-44","trained_model","final_model-2000")
        h_params["model_to_load_file_path"] = model_to_load_file_path
        load_model = True
        skip_training = True
    h_params['load_model'] = load_model
    h_params['skip_training'] = skip_training #skip for visualization, do not skip for curriculum learning/pre-training

    # recursive exponential decay for epsilon
    h_params['e_max'] = e_max = 1.0 #P(random action in at least one joint) = (1 - epsilon)**nJoints
    h_params['e_tau'] = e_tau = max_steps_per_episode * num_episodes * 0.8 /5 # time constant in steps, close to final value at 5 eTau
    addEFactor = 1.0 - (1.0 / e_tau)

    h_params['train_model_steps_period'] = train_model_steps_period = 4 # mnih = 4, period of mini-batch sampling and training
    h_params['update_target_net_rate_tau'] = tau = 0.001 # rate to update target network toward main network
    h_params['learning_rate'] = lrate = 1E-6
    h_params['discount_factor'] = y = 0.99 # mnih:0.99

    with RobotEnv(showGUI, velocity, rewards_normalizer, rewards_decay_rate) as env: 
        tf.reset_default_graph()
        nActionsPerJoint = 3
        h_params['state_size'] = stateSize = env.observation_space_size
        h_params['number_of_actions'] = nActions = env.action_space_size
        h_params['number_of_joints'] = nJoints = nActions // nActionsPerJoint
        mainDQN = DQN(nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate)
        targetDQN = DQN(nActions, stateSize, num_hidden_layers, num_neurons_per_hidden, lrate)

        # save txt file with hyper parameters
        h_params_file_path = os.path.join(current_model_dir_path, "hyper_params.txt")
        with open(h_params_file_path, "w") as h_params_file:
            json.dump(h_params, h_params_file, sort_keys=True, indent=4)

        # initialize and prepare model saving
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables,tau)

        with tf.Session() as sess:
            print("Starting training...")
            start_time = time.time()
            sess.run(init)

            if load_model:
                saver.restore(sess, model_to_load_file_path)
                print('\nPreviously saved model was loaded')

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

            for i in range(1, num_episodes + 1):
                if i % model_saving_period == 0:
                    print("episode number:", i)
                initialState = env.reset() # reset environment and get first observation
                undisc_return = 0
                sum_of_maxQ = np.zeros((nJoints,1))
                done = False
                if not skip_training: episodeBuffer = experience_dataset(replay_memory_size) # temporary buffer
                j = 0
                while j < max_steps_per_episode:
                    # print("\nstep:", j)
                    j += 1
                    total_steps += 1

                    # pick action from the DQN, epsilon greedy
                    chosenActions, allJQValues = sess.run(
                        [mainDQN.allJointsBestActions, mainDQN.allJointsQvalues3D],
                        feed_dict={mainDQN.inState:np.reshape(initialState, (1, nJoints))}
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
                        transition = np.array([initialState, chosenActions, r, newState, done])
                        transition = np.reshape(transition, [1, 5]) # 1 x 5
                        episodeBuffer.add(transition) # add step to episode buffer
                    initialState = newState

                    if total_steps > replay_start_size and not skip_training:
                        # epsilon decay
                        addE *= addEFactor
                        epsilon = e_min + addE
                        if total_steps % train_model_steps_period == 0:
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

                    # end of step, save tracked statistics
                    undisc_return += r
                    maxQvalues = np.reshape(maxQvalues, (nJoints, 1))
                    sum_of_maxQ += maxQvalues
                    if done:
                        success_count +=1
                        break

                #episode ended, save tracked statistics

                # add current episode's list of transitions to dataset
                if not skip_training: dataset.add(episodeBuffer.data)

                num_steps_per_ep.append(j)
                undisc_return_per_ep.append(undisc_return)
                successes.append(success_count)
                epsilon_per_ep.append(epsilon)

                averageMaxQ = sum_of_maxQ / j #nJoints x 1

                print("averageMaxQ for each joint:\n", averageMaxQ.T)

                average_maxQ_per_ep = np.concatenate((average_maxQ_per_ep,averageMaxQ), axis=1)

                #save the model and log training
                if i % model_saving_period ==0:
                    save_path = saver.save(sess, checkpoint_model_file_path, global_step=i)
                    print("\nepisode: {} steps: {} undiscounted return obtained: {} done: {}".format(i, j, undisc_return, done))
                    checkpoints_plots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                    os.makedirs(checkpoints_plots_dir_path, exist_ok=True)
                    savePlots(checkpoints_plots_dir_path, undisc_return_per_ep, num_steps_per_ep,
                              successes, epsilon_per_ep, average_maxQ_per_ep)
            #training ended, save results
            end_time = time.time()
            print("Training ended")

            save_path = saver.save(sess, trained_model_file_path, global_step=num_episodes) #save the trained model
            print("Trained model saved in file: %s" % save_path)

    # time
    total_training_time = end_time - start_time #in seconds
    print('\nTotal training time:', total_training_time)
    time_dict = {"total_training_time_in_secs":total_training_time}

    # save txt file with total time
    total_time_file_path = os.path.join(current_model_dir_path, "total_time.txt")
    with open(total_time_file_path, "w") as total_time_file:
        json.dump(time_dict, total_time_file, sort_keys=True, indent=4)

    ## save plots separately
    savePlots(trained_model_plots_dir_path, undisc_return_per_ep, num_steps_per_ep, successes, epsilon_per_ep, average_maxQ_per_ep)
    #plt.show() #optional