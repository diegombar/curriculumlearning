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

# class Qnetwork():
#     def __init__(self, h_size)


# create folder to save results
current_dir_path = os.path.dirname(os.path.realpath(__file__))
all_models_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()
current_model_dir_path = os.path.join(all_models_dir_path, "model_and_results_" + timestr)
plots_dir_path = os.path.join(current_model_dir_path, "results")
checkpoints_dir_path = os.path.join(current_model_dir_path, "saved_checkpoints")
trained_model_dir_path = os.path.join(current_model_dir_path, "trained_model")

for new_directory in [plots_dir_path, checkpoints_dir_path, trained_model_dir_path]:
    os.makedirs(new_directory, exist_ok=True)

checkpoint_model_file_path = os.path.join(checkpoints_dir_path, "checkpoint_model")
trained_model_file_path = os.path.join(trained_model_dir_path, "final_model")

#params to save in txt file:
params = {}

with RobotEnv() as env:
    # network
    tf.reset_default_graph()
    # feed-forward part to choose actions
    stateSize = env.observation_space_size
    nActions = env.action_space_size
    nHidden = 256          ##########################################TO TUNE?
    lrate = 1E-7 #try 6E-6 ##########################################TO TUNE
    inState = tf.placeholder(shape=[1,stateSize], dtype=tf.float32)

    params['neurons_per_hidden_layer'] = nHidden
    params['number_of_actions'] = nActions
    params['learning_rate'] = lrate

    # initialize params
    W0 = weight_variable([stateSize, nHidden])
    W1 = weight_variable([nHidden, nHidden])
    W2 = weight_variable([nHidden, nActions])
    b0 = bias_variable([1])
    b1 = bias_variable([1])
    b2 = bias_variable([1])

    # layers
    hidden1 = tf.nn.relu(tf.matmul(inState, W0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W1) + b1)
    outQvalues = tf.matmul(hidden2, W2) + b2 # Q values for all actions given inState
    bestAction = tf.argmax(outQvalues, 1) # action with highest Q given inState

    # loss by taking the sum of squares difference between the target and prediction Q values.
    Qtargets = tf.placeholder(shape=[1,nActions], dtype=tf.float32)
    loss = tf.reduce_mean(tf.square(Qtargets - outQvalues))
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    updateModel = optimizer.minimize(loss)

    # initialize and prepare model saving (every 2 hours and maximum 4 latest models)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # Set learning parameters
    y = 0.99 # discount factor
    params['discount_factor'] = y
    num_episodes = 10 #500 # number of runs#########################################TO SET
    params['num_episodes'] = lrate
    max_steps_per_episode = 100 #500 # max number of actions per episode#############TO SET
    params['max_steps_per_episode'] = max_steps_per_episode

    e_max = 1.0 # initial epsilon
    e_min = 0.01 # final epsilon
    num_e_updates = 2 #50 # times e is decreased (has to be =< num_episodes)
    e_update_period = num_episodes // num_e_updates # num of episodes between e updates
    e = e_max #initialize epsilon
    model_saving_period = 5 #50 #episodes
    params['e_max'] = e_max
    params['e_min'] = e_min
    params['num_e_updates'] = num_e_updates
    exp_buffer_size = 1000
    params['exp_buffer_size'] = exp_buffer_size


    #create lists to contain total rewards and steps per episode
    num_steps_per_ep = []
    undisc_return_per_ep = []
    success_count = 0
    successes = []
    avg_maxQ_per_ep = []
    epsilon_per_ep = []

    #experience replay
    dataset = experience_dataset(exp_buffer_size)
    batch_size = 30
    train_model_steps_period = 10
    pre_train_episodes = num_episodes // 10 # num of eps to fill dataset with random actions
    params['batch_size'] = batch_size
    params['train_model_steps_period'] = train_model_steps_period
    params['pre_train_episodes'] = pre_train_episodes

    message = "\nepisode: {} steps: {} undiscounted return obtained: {} done: {}"
    with tf.Session() as sess:
        print("Training the model...")
        start_time = time.time()
        sess.run(init)
        for i in range(1, num_episodes + 1):
            # reset environment and get first new observation
            initialState = env.reset()
            sum_of_r = 0
            sum_of_maxQ = 0
            done = False
            j = 0
            episodeBuffer = experience_dataset(exp_buffer_size) # temporary buffer
            while j < max_steps_per_episode:
                # print("\nstep:", j)
                j+=1
                # pick action from the DQN, epsilon greedy
                chosenAction, allQvalues = sess.run([bestAction,outQvalues], feed_dict={inState:initialState})
                # print("\nchosenAction:", chosenAction)
                # print("\nallQvalues:", allQvalues)
                if np.random.rand(1) < e or i <= pre_train_episodes: chosenAction[0] = np.random.randint(0, nActions-1)

                # perform action and get new state and reward
                newState, r, done = env.step(chosenAction[0])
                # print("\nnewState:", newState)
                print("\nr:", r)
                # print("\ndone:", done)
                transition = np.array([initialState, chosenAction[0], r, newState, done])
                episodeBuffer.add(np.reshape(transition, [1, 5])) # add step to episode buffer

                maxQ = 0
                if i > pre_train_episodes:
                    if j % train_model_steps_period == 0:
                        batch = dataset.sample(batch_size)
                        for s,a,r,s1,d in batch:
                            # get Q values for new state 
                            allNewQvalues = sess.run(outQvalues,feed_dict={inState:s1})
                            # get max for Q target
                            # print("\nallNewQvalues:", allNewQvalues)
                            maxQ = np.max(allNewQvalues)
                            targetQ = allQvalues # dont update Q values corresponding for actions not performed
                            targetQ[0, chosenAction[0]] = r + y * maxQ #only update Q value for action performed
                            #Train our network using target and predicted Q values
                            sess.run([updateModel], feed_dict={inState:s, Qtargets:targetQ})

                sum_of_r += r
                # print("\nmaxQ:", maxQ)
                sum_of_maxQ += maxQ
                initialState = newState

                if done == True:
                    success_count +=1
                    break

            # add current episode's list of transitions to dataset
            dataset.add(episodeBuffer.data)

            # decrease epsilon and save model
            if i % e_update_period == 0:
                e = e - (e_max - e_min)/num_e_updates

            #save the model
            if i % model_saving_period ==0:
                save_path = saver.save(sess, checkpoint_model_file_path, global_step=i)

            # log training progress
            if i % 10 == 0: print(message.format(i, j, sum_of_r, done))

            num_steps_per_ep.append(j)
            undisc_return_per_ep.append(sum_of_r)
            successes.append(success_count)
            averageMaxQ = sum_of_maxQ/j
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
params["total_training_time"] = total_training_time

# save txt file with current parameters
params_file_path = os.path.join(current_model_dir_path, "params.txt")
params_file = open(params_file_path, "w")
json.dump(params, params_file, sort_keys=True, indent=4)
params_file.close()

## save plots separately

fig1 = plt.figure(1)
plt.plot(undisc_return_per_ep)
plt.ylabel('return')
plt.xlabel('episode')
plt.title('Undiscounted return obtained')
returns_file = os.path.join(plots_dir_path, "returns.svg")
fig1.savefig(returns_file, bbox_inches='tight')

# steps performed in each episode
fig2 = plt.figure(2)
plt.plot(num_steps_per_ep)
plt.ylabel('steps')
plt.xlabel('episode')
plt.title('Steps performed per episode')
steps_file = os.path.join(plots_dir_path, "steps.svg")
fig2.savefig(steps_file, bbox_inches='tight')
# percentage of success so far, for each episode
fig3 = plt.figure(3)
plt.plot(successes)
plt.ylabel('successes')
plt.xlabel('episode')
plt.title('Number of successes')
successes_file = os.path.join(plots_dir_path, "successes.svg")
fig3.savefig(successes_file, bbox_inches='tight')

# average (over steps, for each episode) of maxQ
fig4 = plt.figure(4)
plt.plot(avg_maxQ_per_ep)
plt.ylabel('average maxQ')
plt.xlabel('episode number')
plt.title('Average maxQ per episode')
average_q_file = os.path.join(plots_dir_path, "average_q.svg")
fig4.savefig(average_q_file, bbox_inches='tight')

# epsilon updates
fig5 = plt.figure(5)
plt.plot(epsilon_per_ep)
plt.ylabel('epsilon value')
plt.xlabel('episode number')
plt.title('Epsilon updates')
epsilons_file = os.path.join(plots_dir_path, "epsilons.svg")
fig5.savefig(epsilons_file, bbox_inches='tight')

#save params and results in txt file
# plt.suptitle('Total training time: %d s' % total_training_time)

plt.show()

