#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
import numpy as np
import os.path

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

def printlog(functionName, returnCode):
    if returnCode == vrep.simx_return_ok:
        print("{} successful".format(functionName))
    else:
        print("{} got error code: {}".format(functionName, returnCode))

# use with statement
# for example:
# with robotEnv() as robotenv1:
#     ###use robotenv1 here###

home_path = os.path.expanduser('~')

class RobotEnv():
    portNb = 19998 # must match the portNb on server side specified in remoteApiConnections.txt
    vrepPath = os.path.join(home_path, "V-REP_PRO_EDU_V3_4_0_Linux", "vrep.sh")
    #blade "/home/diego/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh"
    #doc lab : "/homes/dam416/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh"
    current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
    scenePath = os.path.join(current_dir_path, "MicoRobot.ttt")

    # initialize the environment
    def __init__(self, showGUI):
        #actions/states/reward/done
        self.action_space_size = 3 * 6 # (+Vel, -Vel, 0) for 6 joints
        self.observation_space_size = 6
        self.action_space = range(0,self.action_space_size)
        self.observation_space = np.zeros((1,self.observation_space_size))
        self.state = np.zeros((1,self.observation_space_size))
        self.reward = 0
        self.goalReached = False
        self.minDistance = 0.01
        #v-rep
        self.vrepProcess = None
        self.clientID = None
        #handles
        self.jointHandles = [0] * 6
        self.fingersH1 = 0
        self.fingersH2 = 0
        self.jointsCollectionHandle = 0
        self.distToGoalHandle = 0
        self.distanceToGoal = None
        self.goal_reward = 1 #reward given at goal
        self.jointVel = 0.3
        self.showGUI = showGUI
        self.distance_decay_rate = 1.0 / 0.3
        self.reward_normalizer = 1.0 / 500.0

    # enter and exit methods: needs with statement (used to exit the v-rep simulation properly)
    def __enter__(self):
        print('Starting environment...')

        # launch v-rep
        if self.showGUI == 0:
            vrep_cmd = [self.vrepPath, '-h', self.scenePath] #  headless mode
        elif self.showGUI == 1:
            vrep_cmd = [self.vrepPath, self.scenePath] #GUI mode
        # elif self.showGUI == 2: #headless mode via ssh
        #     vrep_cmd = "xvfb-run --auto-servernum --server-num=1 /homes/dam416/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh -h -s -q MicoRobot.ttt"
            # vrep_cmd = ['xvfb-run', '--auto-servernum', '--server-num=1', self.vrepPath, '-h', '-s', '-q', self.scenePath]
            # vrep_cmd = ['xvfb-run', '--auto-servernum', '--server-num=1', self.vrepPath, '-h', self.scenePath]

        # NOTE: do not use "stdout=subprocess.PIPE" below to buffer logs, causes deadlock at episode 464! (flushing the buffer may work... but buffering is not needed)
        self.vrepProcess = subprocess.Popen(vrep_cmd, shell=False, preexec_fn=os.setsid)
        # connect to V-Rep Remote Api Server
        vrep.simxFinish(-1) # close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', self.portNb, True, False, 10000, 5) # Connect to V-REP

        if self.clientID == -1:
            print('Failed connecting to remote API server')
        else:
            print('Connected to remote API server')
            ## load scene
            # time.sleep(5) # to avoid errors
            # returnCode = vrep.simxLoadScene(self.clientID, self.scenePath, 1, vrep.simx_opmode_oneshot_wait) # vrep.simx_opmode_blocking is recommended

            # Start simulation
            # vrep.simxSetIntegerSignal(self.clientID, 'dummy', 1, vrep.simx_opmode_blocking)
            returnCode = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
            printlog('simxStartSimulation', returnCode)

            # get handles and start streaming distance to goal
            for i in range(0,6):
                returnCode, self.jointHandles[i] = vrep.simxGetObjectHandle(self.clientID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
            printlog('simxGetObjectHandle', returnCode)
            returnCode, self.fingersH1 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
            returnCode, self.fingersH2 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
            returnCode, self.jointsCollectionHandle = vrep.simxGetCollectionHandle(self.clientID, "sixJoints#", vrep.simx_opmode_blocking)
            returnCode, self.distToGoalHandle = vrep.simxGetDistanceHandle(self.clientID, "distanceToGoal#", vrep.simx_opmode_blocking)
            # returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_streaming) #start streaming
            # returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_streaming) #start streaming

            # get first valid state

            self.updateState()
            ## check the state is valid
            # while True:
            #     self.updateState()
            #     print("\nstate: ", self.state)
            #     print("\nreward: ", self.reward)
            #     # wait for a state
            #     if (self.state.shape == (1,self.observation_space_size) and abs(self.reward) < 1E+5):
            #         print("sent reward3=", self.reward)
            #         break

            print('Environment succesfully initialised, ready for simulations')

            # while vrep.simxGetConnectionId(self.clientID) != -1:  
            #     ##########################EXECUTE ACTIONS AND UPDATE STATE HERE #################################
            #     time.sleep(0.1)
            #     #end of execution loop 

            return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Closing environment...')
        #check simulation is running before stopping (ctrl+C to kill case)
        returnCode, serverState = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
        stopped = not (serverState & 1)
        if not stopped:
            # stop simulation
            returnCode = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
            printlog('simxStopSimulation', returnCode)
        
        # # wait for simulation to stop
        while True:
            returnCode, ping = vrep.simxGetPingTime(self.clientID)
            returnCode, serverState = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
            stopped = not (serverState & 1)
            if stopped:
                print("\nSimulation stopped.")
                break

        # close the scene
        # returnCode = vrep.simxCloseScene(self.clientID,vrep.simx_opmode_blocking)

        #close v-rep
        vrep.simxFinish(self.clientID)
        # self.vrepProcess.terminate() #doesnt work
        os.killpg(os.getpgid(self.vrepProcess.pid), signal.SIGTERM)
        print('Environment successfully closed')
        
    
    # reset the state for each new episode
    def reset(self):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # stop simulation
            returnCode = vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
            if returnCode != vrep.simx_return_ok: print("simxStopSimulation failed, error code:", returnCode)

            # wait for simulation stop
            while True:
                returnCode, ping = vrep.simxGetPingTime(self.clientID)
                returnCode, serverState = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                # printlog('\nsimxGetInMessageInfo', returnCode)
                # print('\nServer state: ', serverState)
                stopped = not (serverState & 1)
                if stopped:
                    print("\nSimulation stopped.")
                    break
            #NOTE: if synchronous mode is needed, check http://www.forum.coppeliarobotics.com/viewtopic.php?f=5&t=6603&sid=7939343e5e04b699af2d46f8d6efe7ba

            #now restart simulation
            returnCode = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)
            # while returnCode != vrep.simx_return_ok:
            #     print("simxStartSimulation failed, error code:", returnCode)
            #     returnCode = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)

            # returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_streaming) #start streaming
            # returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_streaming) #start streaming
            
            # get new measurements
            self.goalReached = False
            self.updateState()
            ## check the state is valid
            # while True:
            #     self.updateState()
            #     print("\nstate: ", self.state)
            #     print("\nreward: ", self.reward)
            #     # wait for a state
            #     if (self.state.shape == (1,self.observation_space_size) and abs(self.reward) < 1E+5):
            #         print("sent reward3=", self.reward)
            #         break

            return self.state

    #update the state
    def updateState(self):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # update joint angles
            # print("Getting state...")
            returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_blocking) # or simx_opmode_blocking (not recommended)
            jointPositions = np.array(floatData[0::2]) #take elements at odd positions (even correspond to torques)
            jointPositions = jointPositions % (2 * np.pi) #convert values to [0, 2*pi[
            newState = [angle if angle <= np.pi else angle - 2 * np.pi for angle in jointPositions] #convert values to ]-pi, +pi]
            unNormalizedState1 = np.array(newState)
            state1 = unNormalizedState1 / np.pi
            # print("New state received")
            try: 
                self.state = state1.reshape((1,6)) #reshape (for tensorflow)
            except:
                pass
            # get reward from distance reading and check goal
            # print("Reading distance...")
            returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_blocking) #dist in metres #vrep.simx_opmode_buffer after streaming start
            # print("Distance received")
            if self.distanceToGoal < self.minDistance:
                self.goalReached = True
                self.reward = self.goal_reward
            else:
                # self.goalReached = False
                self.reward = self.reward_normalizer * np.exp(-self.distance_decay_rate * self.distanceToGoal)

    # execute action
    def step(self, actions):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # 6 * 3 = 18 actions -> action is in [0,17]
            # print("actions", actions)
            # jointNumber = action // 3
            velMode = actions % 3 - 1 # speed to apply -1->-Vel;  0->zero;  +1->+Vel
            # print("velMode", velMode)
            for i in range(0,6):
                returnCode = vrep.simxSetJointTargetVelocity(self.clientID, self.jointHandles[i], velMode[i] * self.jointVel, vrep.simx_opmode_blocking)

            # printlog('simxSetJointTargetVelocity', returnCode)

            ## hand actions
            # def openHand(self.clientID):
            #     closingVel = -0.04
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH1, -closingVel, vrep.simx_opmode_oneshot)
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH2, -closingVel, vrep.simx_opmode_oneshot)

            # def closeHand(self.clientID):
            #     closingVel = -0.04
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH1, closingVel, vrep.simx_opmode_oneshot)
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH2, closingVel, vrep.simx_opmode_oneshot)

            ##set joints target positions
            # def setJointTargetPositions(self.clientID, targetPositions):
            #     for i in range(6):
            #         returnCode = vrep.simxSetJointTargetPosition(self.clientID, jointHandles[i], targetPositions[i], vrep.simx_opmode_oneshot)
            #         if returnCode != vrep.simx_return_ok:
            #             print("SetJointTargetPosition got error code: %s" % returnCode)
            self.updateState()
            ## check the state is valid
            # while True:
            #     self.updateState()
            #     print("\nstate: ", self.state)
            #     print("\nreward: ", self.reward)
            #     # wait for a state
            #     if (self.state.shape == (1,self.observation_space_size) and abs(self.reward) < 1E+5):
            #         print("sent reward3=", self.reward)
            #         break
            return self.state, self.reward, self.goalReached


    