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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def saveRewardFunction(normalizer, decay_rate, dir_path):
    fig = plt.figure()
    d = np.arange(0., 3., 0.05)
    rewards = normalizer * np.exp(- decay_rate * d)
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

def savePlots(dir_path, undisc_returns, num_steps, successes, epsilons, avg_maxQ1s, avg_maxQ2s, avg_maxQ3s, avg_maxQ4s, avg_maxQ5s, avg_maxQ6s):
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
    def __init__(self, nActions, stateSize, nHidden, lrate):
        # stateSize = env.observation_space_size
        # nActions = env.action_space_size
        self.nJoints = 6
        self.inState = tf.placeholder(shape=[None,stateSize], dtype=tf.float32) #batch_size x stateSize
        self.nActionsPerJoint = nActions // self.nJoints

        # initialize params
        self.W0 = weight_variable([stateSize, nHidden])
        self.W1 = weight_variable([nHidden, nHidden])
        self.W2 = weight_variable([nHidden, nActions])

        self.b0 = bias_variable([1])
        self.b1 = bias_variable([1])
        self.b2 = bias_variable([1])

        # layers
        self.hidden1 = tf.nn.relu(tf.matmul(self.inState, self.W0) + self.b0)
        self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W1) + self.b1)
        self.allJointsQvalues = tf.matmul(self.hidden2, self.W2) + self.b2 # Q values for all actions given inState, #batch_size x nActions

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


### create folders to save results ###
current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_model_dir_path = os.path.join(all_models_dir_path, "model_and_results_" + timestr)
trained_model_plots_dir_path = os.path.join(current_model_dir_path, "trained_model_results")
checkpoints_dir_path = os.path.join(current_model_dir_path, "saved_checkpoints")
trained_model_dir_path = os.path.join(current_model_dir_path, "trained_model")
#####
# model_to_load_file_path = "/homes/dam416/curriculumlearning/scripts/v-rep_project/trained_models_and_results/model_and_results_2017-Jul-03_15-24-03/trained_model/final_model-3000"
model_to_load_file_path = "/homes/dam416/curriculumlearning/scripts/v-rep_project/trained_models_and_results/model_and_results_2017-Jul-03_15-24-03/saved_checkpoints/checkpoint_model-400"
h_params["model_to_load_file_path"] = model_to_load_file_path

#####
for new_directory in [trained_model_plots_dir_path, checkpoints_dir_path, trained_model_dir_path]:
    os.makedirs(new_directory, exist_ok=True)

checkpoint_model_file_path = os.path.join(checkpoints_dir_path, "checkpoint_model")
trained_model_file_path = os.path.join(trained_model_dir_path, "final_model")

# Set learning hyper parameters and save them
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
h_params["_commit_hash"] = git_hash.decode("utf-8").strip()

# V-REP params
showGUI = 1
velocity = 0.3
h_params["joint_velocity"] = velocity

#plot and save reward function
rewards_normalizer = 0.1
rewards_decay_rate = 3
h_params['rewards_normalizer'] = rewards_normalizer
h_params['rewards_decay_rate'] = rewards_decay_rate
saveRewardFunction(rewards_normalizer, rewards_decay_rate, current_model_dir_path)

load_model = False ########
skip_training = False #######
h_params['load_model'] = load_model
h_params['skip_training'] = skip_training
with RobotEnv(showGUI, velocity, rewards_normalizer, rewards_decay_rate) as env:
    
    y = 0.99 # discount factor mnih:0.99
    h_params['discount_factor'] = y
    num_episodes = 2000 # number of runs#######################################TO SET
    h_params['num_episodes'] = num_episodes
    max_steps_per_episode = 500 # max number of actions per episode##########TO SET
    h_params['max_steps_per_episode'] = max_steps_per_episode

    e_max = 1.0 # initial epsilon mnih = 1.0
    e_min = 0.01 # final epsilon mnih = 0.0 ##P(random action in at least one joint) = (1 - epsilon)**nJoints
    e_tau = max_steps_per_episode * num_episodes * 0.8 /5 # time constant in steps, close to final value at 5 eTau
    addEFactor = 1.0 - (1.0 / e_tau)

    # e_update_steps = (max_steps_per_episode * num_episodes) // 3  #50 # times e is decreased (has to be =< num_episodes)
    #reach e_min in num_episodes // 2

    model_saving_period = 100  #episodes
    h_params['e_max'] = e_max
    h_params['e_min'] = e_min
    h_params['e_tau'] = e_tau
    # h_params['e_update_steps'] = e_update_steps
    # eDecrease = (e_max - e_min) / e_update_steps

    # eFactor = 1 - 1E-5
    # h_params['e_factor'] = eFactor

    batch_size = 32 #mnih=32
    train_model_steps_period = 4 # mnih = 4
    replay_start_size = 50000 # 100 episodes #num of steps to fill dataset with random actions mnih=5E4
    replay_memory_size = 500000 #1000 #steps #200 ep #mnih: 1E6 about 100 episodes

    # about 50 episodes
    if replay_start_size <= max_steps_per_episode or replay_start_size < batch_size:
        print("WARNING: replay_start_size must be greater than max_steps_per_episode and batch_size")

    h_params['batch_size'] = batch_size
    h_params['train_model_steps_period'] = train_model_steps_period
    h_params['replay_start_size'] = replay_start_size
    h_params['replay_memory_size'] = replay_memory_size

    message = "\nepisode: {} steps: {} undiscounted return obtained: {} done: {}"

    tau = 0.001 #Rate to update target network toward primary network
    h_params['update_target_net_rate_tau'] = tau
    

    h_params['notes'] = "goal_reward = 1, exponential decay reward, normalized angles"

    nHidden = 50 #mnih: 512 for dense hidden layer
    lrate = 1E-6
    h_params['neurons_per_hidden_layer'] = nHidden
    h_params['learning_rate'] = lrate

    #pass 0 for headless mode, 1 to showGUI
    # with RobotEnv(1) as env:
    tf.reset_default_graph()
    stateSize = env.observation_space_size
    nActions = env.action_space_size
    nJoints = 6
    nActionsPerJoint = nActions // nJoints
    h_params['state_size'] = stateSize
    h_params['number_of_actions'] = nActions
    mainDQN = DQN(nActions, stateSize, nHidden, lrate)
    targetDQN = DQN(nActions, stateSize, nHidden, lrate)

    h_params['num_hidden_layers_not_output'] = 4
    h_params['non_linearity'] = "ReLU for hidden layers, none for output"
    h_params['optimizer'] = "Adam"

    # save txt file with current parameters
    h_params_file_path = os.path.join(current_model_dir_path, "hyper_params.txt")
    with open(h_params_file_path, "w") as h_params_file:
        json.dump(h_params, h_params_file, sort_keys=True, indent=4)

    # initialize and prepare model saving (every 2 hours and maximum 4 latest models)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    #create lists to contain total rewards and steps per episode
    num_steps_per_ep = []
    undisc_return_per_ep = []
    success_count = 0
    successes = []
    epsilon_per_ep = []
    avg_maxQ1_per_ep = []
    avg_maxQ2_per_ep = []
    avg_maxQ3_per_ep = []
    avg_maxQ4_per_ep = []
    avg_maxQ5_per_ep = []
    avg_maxQ6_per_ep = []

    #initialize epsilon
    epsilon = e_min
    if not skip_training:
        addE = e_max - e_min
        epsilon = e_min + addE
        #experience replay
        dataset = experience_dataset(replay_memory_size)

    with tf.Session() as sess:
        print("Starting training...")
        start_time = time.time()
        sess.run(init)

        # load trained model/checkpoint, not working for the moment (timestamp in name...)
        if load_model:
            print('Loading Model...')
            saver.restore(sess, model_to_load_file_path)
            print('Model loaded')

        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        total_steps = 0
        for i in range(1, num_episodes + 1):
            if i % model_saving_period ==0:
                print("episode number:", i)
            # reset environment and get first new observation
            initialState = env.reset()
            undisc_return = 0
            sum_of_maxQ1 = 0
            sum_of_maxQ2 = 0
            sum_of_maxQ3 = 0
            sum_of_maxQ4 = 0
            sum_of_maxQ5 = 0
            sum_of_maxQ6 = 0
            done = False
            j = 0

            if not skip_training: episodeBuffer = experience_dataset(replay_memory_size) # temporary buffer

            while j < max_steps_per_episode:
                # print("\nstep:", j)
                j += 1
                total_steps += 1

                # pick action from the DQN, epsilon greedy
                chosenActions, allJQValues = sess.run([mainDQN.allJointsBestActions, mainDQN.allJointsQvalues3D], feed_dict={mainDQN.inState:np.reshape(initialState, (1, nJoints))})

                chosenActions = np.reshape(np.array(chosenActions), nJoints)
                allJQValues = np.reshape(np.array(allJQValues), (nJoints, nActionsPerJoint))
                maxQvalues = allJQValues[range(nJoints), chosenActions] # 1 x nJoints #working

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

                        # bestActionsj1, bestActionsj2, bestActionsj3, bestActionsj4, bestActionsj5, bestActionsj6 = sess.run([mainDQN.bestActionj1, mainDQN.bestActionj2, mainDQN.bestActionj3, mainDQN.bestActionj4, mainDQN.bestActionj5, mainDQN.bestActionj6], feed_dict={mainDQN.inState:states1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a')) #batch_size x 1
                        allJBestActions = sess.run(mainDQN.allJointsBestActions, feed_dict={mainDQN.inState:states1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a')) #batch_size x 1

                        # allQj1, allQj2, allQj3, allQj4, allQj5, allQj6 = sess.run([targetDQN.j1Qvalues, targetDQN.j2Qvalues, targetDQN.j3Qvalues, targetDQN.j4Qvalues, targetDQN.j5Qvalues, targetDQN.j6Qvalues], feed_dict={targetDQN.inState:states1}) #feed btach of s' and get batch of Q2(a') # batch_size x 3
                        allJQvalues = sess.run(targetDQN.allJointsQvalues3D, feed_dict={targetDQN.inState:states1}) #feed btach of s' and get batch of Q2(a') # batch_size x 3

                        #get Q values of best actions
                        allJBestActions_one_hot = np.arange(nActionsPerJoint) == allJBestActions[:,:,None]
                        allJBestActionsQValues = np.sum(np.multiply(allJBestActions_one_hot, allJQvalues), axis=2) # batch_size x nJoints
                        end_multiplier = -(dones - 1) # batch_size x 1
                        end_multiplier = np.reshape(end_multiplier, (batch_size, 1))
                        rewards = np.reshape(rewards, (batch_size, 1))

                        allJQtargets = np.reshape(rewards + y * allJBestActionsQValues * end_multiplier, (batch_size,nJoints)) #batch_size x nJoints

                        #Update the network with our target values.
                        _ = sess.run(mainDQN.updateModel, feed_dict={mainDQN.inState:states0,mainDQN.Qtargets:allJQtargets, mainDQN.chosenActions:actions0})
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

                        # for s,a,r,s1,d in batch: #to change: feed all at once
                        #     # get Q values for new state 
                        #     allNewQvalues = sess.run(outQvalues,feed_dict={inState:s1})
                        #     # get max for Q target
                        #     # print("\nallNewQvalues:", allNewQvalues)
                        #     maxQ = np.max(allNewQvalues)
                        #     targetQ = allJointsQvalues # dont update Q values corresponding for actions not performed
                        #     targetQ[0, chosenAction[0]] = r + y * maxQ #only update Q value for action performed
                        #     #Train our network using target and predicted Q values
                        #     sess.run([updateModel], feed_dict={inState:s, Qtargets:targetQ})

                undisc_return += r
                sum_of_maxQ1 += maxQvalues[0]
                sum_of_maxQ2 += maxQvalues[1]
                sum_of_maxQ3 += maxQvalues[2]
                sum_of_maxQ4 += maxQvalues[3]
                sum_of_maxQ5 += maxQvalues[4]
                sum_of_maxQ6 += maxQvalues[5]

                if done == True:
                    success_count +=1
                    break

            # add current episode's list of transitions to dataset
            if not skip_training: dataset.add(episodeBuffer.data)

            num_steps_per_ep.append(j)
            undisc_return_per_ep.append(undisc_return)
            successes.append(success_count)
            epsilon_per_ep.append(epsilon)

            averageMaxQ1 = sum_of_maxQ1 / j
            averageMaxQ2 = sum_of_maxQ2 / j
            averageMaxQ3 = sum_of_maxQ3 / j
            averageMaxQ4 = sum_of_maxQ4 / j
            averageMaxQ5 = sum_of_maxQ5 / j
            averageMaxQ6 = sum_of_maxQ6 / j

            avg_maxQ1_per_ep.append(averageMaxQ1)
            avg_maxQ2_per_ep.append(averageMaxQ2)
            avg_maxQ3_per_ep.append(averageMaxQ3)
            avg_maxQ4_per_ep.append(averageMaxQ4)
            avg_maxQ5_per_ep.append(averageMaxQ5)
            avg_maxQ6_per_ep.append(averageMaxQ6)

            print("averageMaxQ for each joint: ", averageMaxQ1, averageMaxQ2, averageMaxQ3, averageMaxQ4, averageMaxQ5, averageMaxQ6)
            
            #save the model and log training
            if i % model_saving_period ==0:
                save_path = saver.save(sess, checkpoint_model_file_path, global_step=i)
                print(message.format(i, j, undisc_return, done))
                checkpoints_plots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                os.makedirs(checkpoints_plots_dir_path, exist_ok=True)
                savePlots(checkpoints_plots_dir_path, undisc_return_per_ep, num_steps_per_ep, successes, epsilon_per_ep, avg_maxQ1_per_ep, avg_maxQ2_per_ep, avg_maxQ3_per_ep, avg_maxQ4_per_ep, avg_maxQ5_per_ep, avg_maxQ6_per_ep)

        end_time = time.time()
        print("Training ended")

        #save the trained model
        save_path = saver.save(sess, trained_model_file_path, global_step=num_episodes)
        print("Trained model saved in file: %s" % save_path)

        # #Visualization of learned policy
        # print("press 'v' to start policy test visualization")
        # while True:
        #     c = readchar.readchar()
        #     print('char=',c)
        #     if c == 'v':
        #         break
        # #alternatively, save the weights W0, W1, W2
        # initialState = env.reset()
        # rAll = 0
        # done = False
        # j = 0
        # while j < max_steps_per_episode:
        #     j+=1
        #     chosenAction = sess.run(bestAction,feed_dict={inState:initialState})
        #     newState, r, done = env.step(chosenAction[0])
        #     rAll += r
        #     initialState = newState
        #     if done == True:
        #         break
        # print('\nsteps:', j)
        # print('\nreturn:', rAll)     

# time
total_training_time = end_time - start_time #in seconds
print('\nTotal training time:', total_training_time)
time_dict = {"total_training_time_in_secs":total_training_time}

# save txt file with total time
total_time_file_path = os.path.join(current_model_dir_path, "total_time.txt")
with open(total_time_file_path, "w") as total_time_file:
    json.dump(time_dict, total_time_file, sort_keys=True, indent=4)

## save plots separately
savePlots(trained_model_plots_dir_path, undisc_return_per_ep, num_steps_per_ep, successes, epsilon_per_ep, avg_maxQ1_per_ep, avg_maxQ2_per_ep, avg_maxQ3_per_ep, avg_maxQ4_per_ep, avg_maxQ5_per_ep, avg_maxQ6_per_ep)
#plt.show() #optional

