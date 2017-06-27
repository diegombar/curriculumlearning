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

# Load environment
from robotenv import RobotEnv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def savePlot(var_value_per_ep, ylabel_str, title_str, dir_path, name):
    fig = plt.figure()
    plt.plot(var_value_per_ep)
    plt.ylabel(ylabel_str)
    plt.xlabel('episode')
    plt.title(title_str)
    plot_file = os.path.join(dir_path, name)
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close()

def savePlots(disc_returns, num_steps, successes, avg_maxQs, epsilons, dir_path):
    #note: "per_ep" in variable names were omitted
    # discounted returns for each episode
    savePlot(disc_returns, "disc. return", "Discounted return obtained", dir_path, "disc_returns.svg")

    # steps performed in each episode
    savePlot(num_steps, "steps", "Steps performed per episode", dir_path, "steps.svg")

    # number of success so far, for each episode
    savePlot(successes, "successes", "Number of successes", dir_path, "successes.svg")
    # average (over steps, for each episode) of maxQ
    savePlot(avg_maxQs, "average maxQ", "Average maxQ per episode", dir_path, "average_q.svg")
    # epsilon evolution
    savePlot(epsilons, "epsilon value", "Epsilon updates", dir_path, "epsilons.svg")

# experience replay dataset, experience = (s,a,r,s',done)
class experience_dataset():
    def __init__(self, size):
        self.data = []
        self.size = size

    #add experience = list of transitions, freeing space if needed
    def add(self, experience):
        excess = len(self.data) + len(experience) - self.size
        if excess > 0: self.data[0:excess] = []
        self.data.extend(experience) 

    # randomly sample an array of transitions (s,a,r,s',done)
    def sample(self,sample_size):
        sample = np.array(random.sample(self.data,sample_size))
        return np.reshape(sample, [sample_size,5])

class DQN():
    def __init__(self, nActions, stateSize):
        # stateSize = env.observation_space_size
        # nActions = env.action_space_size
        nHidden = 512 #mnih: 512 for dense hidden layer
        lrate = 1E-6 
        h_params['neurons_per_hidden_layer'] = nHidden
        h_params['number_of_actions'] = nActions
        h_params['learning_rate'] = lrate

        self.inState = tf.placeholder(shape=[None,stateSize], dtype=tf.float32) #batch_size x stateSize
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
        self.allQvalues = tf.matmul(self.hidden2, self.W2) + self.b2 # Q values for all actions given inState, #batch_size x nActions
        h_params['num_hidden_layers_not_output'] = 2
        h_params['non_linearity'] = "ReLU for hidden layers, none for output"


        # get batch of chosen actions a0
        self.chosenActions = tf.placeholder(shape=[None],dtype=tf.int32) #batch_size x 1
        #select Q values for chosen actions a0
        self.chosenAs_onehot = tf.one_hot(self.chosenActions,nActions,dtype=tf.float32)
        self.QChosenActions = tf.reduce_sum(tf.multiply(self.allQvalues, self.chosenAs_onehot), axis=1)###

        # action with highest Q given inState
        self.bestAction = tf.argmax(self.allQvalues, 1) #batch_size x 1

        # loss by taking the sum of squares difference between the target and prediction Q values.
        self.Qtargets = tf.placeholder(shape=[None], dtype=tf.float32) #batch_size x 1
        self.error = tf.square(self.Qtargets - self.QChosenActions)
        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
        h_params['optimizer'] = "Adam"
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
current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_model_dir_path = os.path.join(all_models_dir_path, "model_and_results_" + timestr)
trained_model_plots_dir_path = os.path.join(current_model_dir_path, "trained_model_results")
checkpoints_dir_path = os.path.join(current_model_dir_path, "saved_checkpoints")
trained_model_dir_path = os.path.join(current_model_dir_path, "trained_model")

for new_directory in [trained_model_plots_dir_path, checkpoints_dir_path, trained_model_dir_path]:
    os.makedirs(new_directory, exist_ok=True)

checkpoint_model_file_path = os.path.join(checkpoints_dir_path, "checkpoint_model")
trained_model_file_path = os.path.join(trained_model_dir_path, "final_model")
h_params = {} # params to save in txt file:

# Set learning parameters
y = 0.99 # discount factor mnih:0.99
h_params['discount_factor'] = y
num_episodes = 2000 # number of runs#######################################TO SET
h_params['num_episodes'] = num_episodes
max_steps_per_episode = 1000 # max number of actions per episode##########TO SET
h_params['max_steps_per_episode'] = max_steps_per_episode

e_max = 1.0 # initial epsilon mnih = 1.0
e_min = 0.01 # final epsilon mnih = 0.01
e_update_steps = (max_steps_per_episode * num_episodes) // 3  #50 # times e is decreased (has to be =< num_episodes)
#reach e_min in num_episodes // 2
e = e_max #initialize epsilon
model_saving_period = 100 #episodes
h_params['e_max'] = e_max
h_params['e_min'] = e_min
h_params['e_update_steps'] = e_update_steps
eDecrease = (e_max - e_min) / e_update_steps
replay_memory_size = 100000 #mnih: 1E6 about 100 episodes
h_params['replay_memory_size'] = replay_memory_size

# eFactor = 1 - 1E-5
# h_params['e_factor'] = eFactor

#experience replay
dataset = experience_dataset(replay_memory_size)
batch_size = 32 #mnih=32
train_model_steps_period = 4 # mnih = 4
replay_start_size = 50000 # num of steps to fill dataset with random actions mnih=5E4
# about 50 episodes
if replay_start_size <= max_steps_per_episode or replay_start_size < batch_size:
    print("WARNING: replay_start_size must be greater than max_steps_per_episode and batch_size")

h_params['batch_size'] = batch_size
h_params['train_model_steps_period'] = train_model_steps_period
h_params['replay_start_size'] = replay_start_size

message = "\nepisode: {} steps: {} undiscounted return obtained: {} done: {}"

tau = 0.001 #Rate to update target network toward primary network
h_params['update_target_net_rate_tau'] = tau
load_model = "false" # "false", "trained", "checkpoint"

# save txt file with current parameters
h_params_file_path = os.path.join(current_model_dir_path, "hyper_params.txt")
with open(h_params_file_path, "w") as h_params_file:
    json.dump(h_params, h_params_file, sort_keys=True, indent=4)

#pass 0 for headless mode, 1 to showGUI
with RobotEnv(1) as env:
    tf.reset_default_graph()
    stateSize = env.observation_space_size
    nActions = env.action_space_size
    mainDQN = DQN(nActions, stateSize)
    targetDQN = DQN(nActions, stateSize)

    # initialize and prepare model saving (every 2 hours and maximum 4 latest models)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    #create lists to contain total rewards and steps per episode
    num_steps_per_ep = []
    disc_return_per_ep = []
    success_count = 0
    successes = []
    avg_maxQ_per_ep = []
    epsilon_per_ep = []

    with tf.Session() as sess:
        print("Starting training...")
        start_time = time.time()
        sess.run(init)

        #load trained model/checkpoint, not working for the moment (timestamp in name...)
        # if load_model != "false":
        #     print('Loading Model...')
        #     if load_model =="trained":
        #         path = checkpoint_model_file_path
        #     elif load_model =="checkpoint":
        #         path = trained_model_file_path
        #     ckpt = tf.train.get_checkpoint_state(path)
        #     saver.restore(sess,ckpt.model_checkpoint_path)
        #     print('Model loaded')

        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        total_steps = 0
        for i in range(1, num_episodes + 1):
            # reset environment and get first new observation
            initialState = env.reset()
            disc_return = 0
            sum_of_maxQ = 0
            done = False
            j = 0
            episodeBuffer = experience_dataset(replay_memory_size) # temporary buffer

            while j < max_steps_per_episode:
                # print("\nstep:", j)
                j += 1
                total_steps += 1

                # pick action from the DQN, epsilon greedy
                chosenAction, allQvalues = sess.run([mainDQN.bestAction, mainDQN.allQvalues], feed_dict={mainDQN.inState:initialState})
                # print("\nchosenAction:", chosenAction)
                maxQ = np.max(allQvalues)
                # print("\nallQvalues:", allQvalues)
                if np.random.rand(1) < e or total_steps <= replay_start_size:
                    chosenAction[0] = np.random.randint(0, nActions-1)

                # perform action and get new state and reward
                newState, r, done = env.step(chosenAction[0])
                # print("\nnewState:", newState)
                # print("\ndone:", done)
                transition = np.array([initialState, chosenAction[0], r, newState, done])
                episodeBuffer.add(np.reshape(transition, [1, 5])) # add step to episode buffer

                if total_steps > replay_start_size:
                    # decrease epsilon
                    if e > e_min:
                        e -= eDecrease

                    if total_steps % train_model_steps_period == 0:
                        
                        batch = dataset.sample(batch_size)
                        batchOfStates0 = np.vstack(batch[:,0])
                        batchOfActions0 = batch[:,1]
                        batchOfRewards = batch[:,2]
                        batchOfStates1 = np.vstack(batch[:,3])
                        batchOfDones = batch[:,4]
                        batchOfbestActions = sess.run(mainDQN.bestAction,feed_dict={mainDQN.inState:batchOfStates1}) #feed batch of s' and get batch of a' = argmax(Q1(s',a'))
                        batchOfQForAllActions = sess.run(targetDQN.allQvalues,feed_dict={targetDQN.inState:batchOfStates1}) #feed btach of s' and get batch of Q2(a')

                        #get Q values of best actions
                        batchOfQForBestActions = batchOfQForAllActions[range(batch_size), batchOfbestActions]
                        end_multiplier = -(batchOfDones - 1)
                        targetQ =  batchOfRewards + y * batchOfQForBestActions * end_multiplier

                        #Update the network with our target values.
                        _ = sess.run(mainDQN.updateModel, feed_dict={mainDQN.inState:batchOfStates0,mainDQN.Qtargets:targetQ, mainDQN.chosenActions:batchOfActions0})
                        
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

                        # for s,a,r,s1,d in batch: #to change: feed all at once
                        #     # get Q values for new state 
                        #     allNewQvalues = sess.run(outQvalues,feed_dict={inState:s1})
                        #     # get max for Q target
                        #     # print("\nallNewQvalues:", allNewQvalues)
                        #     maxQ = np.max(allNewQvalues)
                        #     targetQ = allQvalues # dont update Q values corresponding for actions not performed
                        #     targetQ[0, chosenAction[0]] = r + y * maxQ #only update Q value for action performed
                        #     #Train our network using target and predicted Q values
                        #     sess.run([updateModel], feed_dict={inState:s, Qtargets:targetQ})

                disc_return = r + y * disc_return 
                # print("\nmaxQ:", maxQ)
                sum_of_maxQ += maxQ
                initialState = newState

                if done == True:
                    success_count +=1
                    break

            # add current episode's list of transitions to dataset
            dataset.add(episodeBuffer.data)

            #save the model and log training
            if i % model_saving_period ==0:
                print("\nr:", r) ############ print if having problems
                save_path = saver.save(sess, checkpoint_model_file_path, global_step=i)
                training_time = time.time() - start_time #in seconds
                print('\nCurrent training time:', training_time)
                print(message.format(i, j, disc_return, done))
                print("Saved Model")
                checkpoints_plots_dir_path = os.path.join(current_model_dir_path, "checkpoint_results_ep_" + str(i))
                os.makedirs(checkpoints_plots_dir_path, exist_ok=True)
                savePlots(disc_return_per_ep, num_steps_per_ep, successes, avg_maxQ_per_ep, epsilon_per_ep, checkpoints_plots_dir_path)
                print("Saved Plots")                

            if i % (model_saving_period // 10) ==0:
                print("episode number:", i)

            num_steps_per_ep.append(j)
            disc_return_per_ep.append(disc_return)
            successes.append(success_count)
            averageMaxQ = sum_of_maxQ / j
            print("averageMaxQ:", averageMaxQ)
            avg_maxQ_per_ep.append(averageMaxQ)
            epsilon_per_ep.append(e)
        
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

# statistics
# print("Percent of succesful episodes: " + str(sum(undisc_return_per_ep)/num_episodes) + "%")

# time
total_training_time = end_time - start_time #in seconds
print('\nTotal training time:', total_training_time)
time_dict = {"total_training_time_in_secs":total_training_time}

# save txt file with current parameters
total_time_file_path = os.path.join(current_model_dir_path, "total_time.txt")
with open(total_time_file_path, "w") as total_time_file:
    json.dump(time_dict, total_time_file, sort_keys=True, indent=4)

#save txt file with total time

## save plots separately
savePlots(disc_return_per_ep, num_steps_per_ep, successes, avg_maxQ_per_ep, epsilon_per_ep, trained_model_plots_dir_path)

#plt.show() #optional

