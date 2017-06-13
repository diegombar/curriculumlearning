#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
import readchar
import numpy as np

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

# #example
    # with robotEnv() as robotenv1:
    #     #use robotenv1

class RobotEnv():
    portNb = 19998 # must match the portNb on server side specified in remoteApiConnections.txt
    vrepPath = "/home/diego/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh"
    scenePath = 'MicoRobot.ttt'
    # initialize the environment
    def __init__(self):
        #actions/states/reward/done
        self.action_space_size = 3 * 6 # (+Vel, -Vel, 0) for 6 joints
        self.observation_space_size = 6
        self.action_space = range(0,self.action_space_size)
        self.observation_space = np.zeros((1,self.observation_space_size))
        self.state = None
        self.reward = None
        self.goalReached = False
        self.minDistance = 0.01
        #v-rep
        self.vrepProcess = None
        self.clientID = None
        #handles
        self.jointHandles = None
        self.fingersH1 = None
        self.fingersH2 = None
        self.jointsCollectionHandle = None
        self.distToGoalHandle = None
        self.distanceToGoal = None

    #   needs with statement (used to exit the v-rep simulation properly)
    def __enter__(self):
        print('Creating environment...')
        # launch v-rep
        self.vrepProcess = subprocess.Popen(self.vrepPath, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid)
        # connect to V-Rep Remote Api Server
        vrep.simxFinish(-1)# close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', self.portNb, True, False, 5000, 5) # Connect to V-REP

        if self.clientID == -1:
            print('Failed connecting to remote API server')
        else:
            print('Connected to remote API server')
            # load scene
            time.sleep(5) # to avoid errors
            returnCode = vrep.simxLoadScene(self.clientID, self.scenePath, 1, vrep.simx_opmode_oneshot_wait) # vrep.simx_opmode_blocking is recommended
            printlog('simxLoadScene', returnCode)

            # args
            self.action_space_size = 3 * 6 # (+Vel, -Vel, 0) for 6 joints
            self.observation_space_size = 6
            self.action_space = range(0,self.action_space_size)
            self.observation_space = np.zeros((1,self.observation_space_size))
            self.goalReached = False
            self.minDistance = 0.01 #one cm from goal

            # Start simulation
            returnCode = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
            printlog('simxStartSimulation', returnCode)

            # get handles and start streaming distance to goal
            self.jointHandles = [0] * 6
            for i in range(0,6):
                returnCode, self.jointHandles[i] = vrep.simxGetObjectHandle(self.clientID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
                if i==0: printlog('simxGetObjectHandle', returnCode)
            returnCode, self.fingersH1 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
            returnCode, self.fingersH2 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
            returnCode, self.jointsCollectionHandle = vrep.simxGetCollectionHandle(self.clientID, "sixJoints#", vrep.simx_opmode_blocking)
            returnCode, self.distToGoalHandle = vrep.simxGetDistanceHandle(self.clientID, "distanceToGoal#", vrep.simx_opmode_blocking)
            returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_streaming) #start streaming
            returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_streaming) #start streaming
            # get first state
            self.updateState()
            
            # # check server state before loop
            # while vrep.simxGetConnectionId(self.clientID) != -1:  
            #     ##########################EXECUTE ACTIONS AND UPDATE STATE HERE #################################
            #     time.sleep(0.1)
            #     #end of execution loop
            print('Environment succesfully initialised, ready for simulations')
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Closing environment...')
        # stop simulation
        returnCode = vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
        printlog('simxStopSimulation', returnCode)
        # close the scene
        running = True
        while running:
            returnCode, ping = vrep.simxGetPingTime(self.clientID)
            returnCode, serverState = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
            running = serverState
        returnCode = vrep.simxCloseScene(self.clientID,vrep.simx_opmode_blocking)
        printlog('simxCloseScene', returnCode)
        #close v-rep
        vrep.simxFinish(self.clientID)
        os.killpg(os.getpgid(self.vrepProcess.pid), signal.SIGTERM)
        print('Environment successfully closed')
        
    
    # reset the state for each new episode
    def reset(self):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # reset robot position in V-REP, i.e. reset simulation
            # stop simulation
            returnCode = vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
            printlog('simxStopSimulation', returnCode)
            running = True
            while running:
                returnCode, ping = vrep.simxGetPingTime(self.clientID)
                returnCode, serverState = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                running = serverState
            # Start simulation
            returnCode = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)
            printlog('simxStartSimulation', returnCode)
            self.updateState()
            return self.state

    #update the state
    def updateState(self):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # update joint angles
            returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_buffer) # or simx_opmode_blocking (not recommended)
            jointPositions = np.array(floatData[0::2]) #take elements at odd positions (even corresponds to torques)
            jointPositions = jointPositions % (2 * np.pi) #take values in [0, 2*pi[
            self.state = [angle if angle <= np.pi else angle - 2 * np.pi for angle in jointPositions] #take values in ]-pi, +pi]
            # get reward from distance reading and check goal
            returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_buffer) #dist in metres        
            if self.distanceToGoal < self.minDistance:
                self.goalReached = True
                self.reward = 10
            else:
                self.goalReached = False
                self.reward = -self.distanceToGoal

    # execute action
    def step(self, action):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # 6 * 3 = 18 actions -> action is in [0,17]
            Vel = 1
            jointNumber = action // 3
            velMode = action % 3 - 1 # speed to apply -1->-Vel;  0->zero;  +1->+Vel
            returnCode = vrep.simxSetJointTargetVelocity(self.clientID, self.jointHandles[jointNumber], velMode * Vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
            ## hand acitons
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
            return self.state, self.reward, self.goalReached


    