# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
# import random
import tensorflow as tf
from matplotlib import pyplot as plt
# import readchar
import time


# Load environment
from robotenv import RobotEnv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

with RobotEnv() as env:
    path_to_saves_folder = "/home/diego/curriculumlearning/scripts/v-rep_project/trained_models_and_results/"
    model_folder_name = "modelTest1/"
    path_to_model_file = path_to_saves_folder + model_folder_name + "model"
    path_to_trained_model_file = path_to_saves_folder + model_folder_name + "trained_model"
    path_to_img_file = path_to_saves_folder + model_folder_name + "results.svg"

    # network
    tf.reset_default_graph()
    # feed-forward part to choose actions
    stateSize = env.observation_space_size
    nActions = env.action_space_size
    nhidden = 256          ##########################################TO TUNE?
    lrate = 1E-5 #try 6E-6 ##########################################TO TUNE
    inState = tf.placeholder(shape=[1,stateSize], dtype=tf.float32)

    # initialize params
    W0 = weight_variable([stateSize, nhidden])
    W1 = weight_variable([nhidden, nhidden])
    W2 = weight_variable([nhidden, nActions])
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
    loss = tf.reduce_sum(tf.square(Qtargets - outQvalues))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate) #try Adam? 
    updateModel = optimizer.minimize(loss)

    # initialize and prepare model saving (every 2 hours and maximum 4 latest models)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # Set learning parameters
    y = 0.99 # discount factor
    
    num_episodes = 500 # number of runs#########################################TO SET

    max_steps_per_episode = 500 # max number of actions per episode#############TO SET

    e_max = 1.0 # initial epsilon
    e_min = 0.01 # final epsilon
    num_e_updates = 50 # times e is decreased (has to be < num_episodes)
    e_update_period = num_episodes // num_e_updates # num of episodes between e updates
    e = e_max #initialize epsilon
    model_saving_period = 50 #episodes

    #create lists to contain total rewards and steps per episode
    num_steps_per_ep = []
    undisc_return_per_ep = []
    success_count = 0
    successes = []
    avg_maxQ_per_ep = []
    epsilon_per_ep = []
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
            while j < max_steps_per_episode:
                j+=1
                # pick action from the DQN, epsilon greedy
                chosenAction, allQvalues = sess.run([bestAction,outQvalues], feed_dict={inState:initialState})
                if np.random.rand(1) < e: chosenAction[0] = np.random.randint(0, nActions-1)

                # perform action and get new state and reward
                newState, r, done = env.step(chosenAction[0])

                # get Q values for new state 
                allNewQvalues = sess.run(outQvalues,feed_dict={inState:newState})
                # get max for Q target
                maxQ = np.max(allNewQvalues)
                targetQ = allQvalues # dont update Q values corresponding for actions not performed
                targetQ[0, chosenAction[0]] = r + y * maxQ #only update Q value for action performed
                #Train our network using target and predicted Q values
                sess.run([updateModel], feed_dict={inState:initialState, Qtargets:targetQ})
                sum_of_r += r
                sum_of_maxQ += maxQ
                initialState = newState

                if done == True:
                    success_count +=1
                    break

            # decrease epsilon and save model
            if i % e_update_period == 0:
                e = e - (e_max - e_min)/num_e_updates

            #save the model
            if i % model_saving_period ==0:
                save_path = saver.save(sess, path_to_model_file, global_step=i)

            # log training progress
            if i % 10 == 0: print(message.format(i, j, sum_of_r, done))

            num_steps_per_ep.append(j)
            undisc_return_per_ep.append(sum_of_r)
            successes.append(success_count)
            avg_maxQ_per_ep.append(sum_of_maxQ/j)
            epsilon_per_ep.append(e)
        
        end_time = time.time()
        print("Training ended")

        #save the trained model
        save_path = saver.save(sess, path_to_trained_model_file, global_step=num_episodes)
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

# total return observed in each episode
fig = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(undisc_return_per_ep)
plt.ylabel('return')
plt.xlabel('episode')
plt.title('Undiscounted return obtained')

# steps performed in each episode
plt.subplot(3,2,2)
plt.plot(num_steps_per_ep)
plt.ylabel('steps')
plt.xlabel('episode')
plt.title('Steps performed per episode')

# percentage of success so far, for each episode
plt.subplot(3,2,3)
plt.plot(successes)
plt.ylabel('successes')
plt.xlabel('episode')
plt.title('Number of successes')

# average (over steps, for each episode) of maxQ
plt.subplot(3,2,4)
plt.plot(avg_maxQ_per_ep)
plt.ylabel('average maxQ')
plt.xlabel('episode number')
plt.title('Average maxQ per episode')

# epsilon updates
plt.subplot(3,2,5)
plt.plot(epsilon_per_ep)
plt.ylabel('epsilon value')
plt.xlabel('episode number')
plt.title('Epsilon updates')

plt.suptitle('Total training time: %d s' % total_training_time)

fig.subplots_adjust(hspace=0.5, wspace=0.5)

plt.savefig(path_to_img_file) # bbox_inches='tight'
# plt.show()
