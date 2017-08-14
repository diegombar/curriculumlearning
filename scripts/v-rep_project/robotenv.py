#!/usr/bin/env python3
import os,signal,subprocess,time
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

# use with statement
# for example:
# with robotEnv() as robotenv1:
#     ###use robotenv1 here###

home_path = os.path.expanduser('~')

#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

class RobotEnv():
    # portNb = 19998
    vrepPath = os.path.join(home_path, "V-REP_PRO_EDU_V3_4_0_Linux", "vrep.sh")
    #blade "/home/diego/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh"
    #doc lab : "/homes/dam416/V-REP_PRO_EDU_V3_3_1_64_Linux/vrep.sh"
    current_dir_path = os.path.dirname(os.path.realpath(__file__)) # directory of this .py file
    scenePath = os.path.join(current_dir_path, "MicoRobot.ttt")
    # scenePath = os.path.join(current_dir_path, "mico_scene_vrep3-3-1.ttt")

    # initialize the environment
    def __init__(self,
                 task,
                 targetPosition,
                 rewards_normalizer,
                 rewards_decay_rate,
                 showGUI=True,
                 velocity=1,
                 nSJoints=6,
                 nAJoints=6,
                 portNb = 19998
                 ):
        #actions/states/reward/done
        self.task = task #see tasks 1, 2 above
        self.nSJoints = nSJoints  #num of joints to include in state vector (starts at base)
        self.nAJoints = nAJoints  #num of actionable joints (starts at base)
        self.portNb = portNb
        
        if self.task == TASK_REACH_CUBE:
            self.observation_space_size = self.nSJoints  # 6 joint angles
        elif self.task == TASK_PUSH_CUBE_TO_TARGET_POSITION:
            self.targetPosition = targetPosition #tuple (x,y) target position relative to robot base
            self.observation_space_size = self.nSJoints + 2  # FOR NOW #8 # 6 joint angles, cube.x, cube.y
        self.action_space_size = 3 * self.nAJoints  # (+Vel, -Vel, 0) for 6 joints
        self.action_space = range(0,self.action_space_size)
        self.observation_space = np.zeros((1,self.observation_space_size))
        self.state = np.zeros((1,self.observation_space_size))
        self.reward = 0
        self.goalReached = False
        self.minDistance = 0.01  #1 cm
        #v-rep
        self.vrepProcess = None
        self.clientID = None
        #handles
        self.jointHandles = [0] * self.nSJoints
        self.fingersH1 = 0
        self.fingersH2 = 0
        self.jointsCollectionHandle = 0
        self.distToGoalHandle = 0
        self.distanceToGoal = None
        # self.goal_reward = 1 #reward given at goal
        self.jointVel = velocity
        self.showGUI = showGUI
        self.distance_decay_rate = rewards_decay_rate  #=1/0.3, reward is close to zero for 5 x 0.3 = 1.5 m
        self.reward_normalizer = rewards_normalizer

    # 'with' statement (used to exit the v-rep simulation properly)
    def __enter__(self):
        print('Starting environment...')

        # launch v-rep
        vrep_cmd = [self.vrepPath, '-gREMOTEAPISERVERSERVICE_' + str(self.portNb) + '_FALSE_FALSE']
        if not self.showGUI:
            vrep_cmd.append('-h')  #headless mode
        vrep_cmd.append(self.scenePath)

        #headless mode via ssh
        #     vrep_cmd = "xvfb-run --auto-servernum --server-num=1 /homes/dam416/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh -h -s -q MicoRobot.ttt"
            # vrep_cmd = ['xvfb-run', '--auto-servernum', '--server-num=1', self.vrepPath, '-h', '-s', '-q', self.scenePath]
            # vrep_cmd = ['xvfb-run', '--auto-servernum', '--server-num=1', self.vrepPath, '-h', self.scenePath]
        print('Launching V-REP...')
        # NOTE: do not use "stdout=subprocess.PIPE" below to buffer logs, causes deadlock at episode 464! (flushing the buffer may work... but buffering is not needed)
        self.vrepProcess = subprocess.Popen(vrep_cmd, shell=False, preexec_fn=os.setsid)
        # connect to V-Rep Remote Api Server
        vrep.simxFinish(-1) # close all opened connections
        # Connect to V-REP
        print('Connecting to V-REP...')
        counter = 0
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', self.portNb, True, False, 5000, 0)
            if self.clientID != -1:
                break
            else:
                print("connection failed, retrying")
                counter += 1
                if counter == 10:
                    raise RuntimeError('Connection to V-REP failed.')



        if self.clientID == -1:
            print('Failed connecting to remote API server')
        else:
            print('Connected to remote API server')

            ##close model browser and hierarchy window
            vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_browser_visible, False, vrep.simx_opmode_blocking)
            vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_hierarchy_visible, False, vrep.simx_opmode_blocking)
            vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_console_visible, False, vrep.simx_opmode_blocking)

            ## load scene
            # time.sleep(5) # to avoid errors
            # returnCode = vrep.simxLoadScene(self.clientID, self.scenePath, 1, vrep.simx_opmode_oneshot_wait) # vrep.simx_opmode_blocking is recommended

            # Start simulation
            # vrep.simxSetIntegerSignal(self.clientID, 'dummy', 1, vrep.simx_opmode_blocking)
            time.sleep(5)  #to center window for recordings
            returnCode = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
            printlog('simxStartSimulation', returnCode)

            # get handles and start streaming distance to goal
            for i in range(0, self.nSJoints):
                returnCode, self.jointHandles[i] = vrep.simxGetObjectHandle(self.clientID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
            printlog('simxGetObjectHandle', returnCode)
            returnCode, self.fingersH1 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
            returnCode, self.fingersH2 = vrep.simxGetObjectHandle(self.clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
            returnCode, self.goalCube = vrep.simxGetObjectHandle(self.clientID, 'GoalCube', vrep.simx_opmode_blocking)
            returnCode, self.robotBase = vrep.simxGetObjectHandle(self.clientID, 'Mico_link1_visible', vrep.simx_opmode_blocking)
            returnCode, self.jointsCollectionHandle = vrep.simxGetCollectionHandle(self.clientID, "sixJoints#", vrep.simx_opmode_blocking)
            returnCode, self.distToGoalHandle = vrep.simxGetDistanceHandle(self.clientID, "distanceToGoal#", vrep.simx_opmode_blocking)
            # returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_streaming) #start streaming
            # returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_streaming) #start streaming

            # get first valid state

            self.updateState() # default initial state: 180 degrees (=pi radians) for all angles
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

    def distance2reward(self, distance):
        return self.reward_normalizer * np.exp(-self.distance_decay_rate * distance)

    #update the state
    def updateState(self,centerInZeroRad=False):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # NOTE: default initial state: 180 degrees (=pi radians) for all angles
            # update joint angles, normalize to ]0,1]
            returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(self.clientID, self.jointsCollectionHandle, 15, vrep.simx_opmode_blocking) # or simx_opmode_blocking (not recommended)
            jointPositions = np.array(floatData[0::2]) #take elements at odd positions (even correspond to torques)
            jointPositions = jointPositions % (2 * np.pi) #convert values to [0, 2*pi[

            if centerInZeroRad:
                newJPositions = [angle if angle <= np.pi else angle - 2 * np.pi for angle in jointPositions] # convert values to ]-pi, +pi]
                newJPositions = np.array(newJPositions)
                newJPositions = newJPositions + np.pi # convert values to ]0, +2*pi]
            else:
                #center in pi
                newJPositions = np.array(jointPositions)


            # newJPositions2 = newJPositions / np.pi # previous version (mistake), convert to ]0, 2]
            newJPositions2 = newJPositions / (2 * np.pi)  # convert to ]0, 1]
            # try:
            newJPositions3 = newJPositions2.reshape(6)
            # except:
            #     pass
            # get reward from distance reading and check goal
            # print("Reading distance...")

            # select the first relevant joints
            self.state = newJPositions3[0:self.nSJoints]

            returnCode, goalCubeRelPos = vrep.simxGetObjectPosition(self.clientID, self.goalCube, self.robotBase, vrep.simx_opmode_blocking)
            x, y, z = goalCubeRelPos  #works, z doesnt change in the plane

            if self.task == TASK_REACH_CUBE:
                returnCode, self.distanceToGoal = vrep.simxReadDistance(self.clientID, self.distToGoalHandle, vrep.simx_opmode_blocking) #dist in metres #vrep.simx_opmode_buffer after streaming start
                self.reward = self.distance2reward(self.distanceToGoal)
            elif self.task == TASK_PUSH_CUBE_TO_TARGET_POSITION:
                target_x, target_y = self.targetPosition
                self.state = np.concatenate(self.state, [x, y]) #add object position
                self.distanceToGoal = np.sqrt((target_x - x)**2 + (target_y - y)**2)
                print('\n Distance to Target Position: ', self.distanceToGoal)
            else:
                print('ERROR: Invalid Task')

            if self.distanceToGoal < self.minDistance:
                self.goalReached = True
                print('#### SUCCESS ####')

    # execute action ACTIONS SHOULD BE THE RIGHT SIZE
    def step(self, actions):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            # 6 * 3 = 18 actions -> action is in [0,17]
            # print("actions", actions)
            # jointNumber = action // 3
            velMode = actions % 3 - 1  # speed to apply -1->-Vel;  0->zero;  +1->+Vel
            # print("velMode", velMode)
            for i in range(0, self.nAJoints):
                returnCode = vrep.simxSetJointTargetVelocity(self.clientID, self.jointHandles[i], velMode[i] * self.jointVel, vrep.simx_opmode_blocking)

            # printlog('simxSetJointTargetVelocity', returnCode)

            ## hand actions
            # def openHand(self.clientID):
            #     closingVel = -0.04
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH1, -closingVel, vrep.simx_opmode_blocking)
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH2, -closingVel, vrep.simx_opmode_blocking)

            # def closeHand(self.clientID):
            #     closingVel = -0.04
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH1, closingVel, vrep.simx_opmode_blocking)
            #     returnCode = vrep.simxSetJointTargetVelocity(self.clientID, fingersH2, closingVel, vrep.simx_opmode_blocking)

            ##set joints target positions
            # def setJointTargetPositions(self.clientID, targetPositions):
            #     for i in range(6):
            #         returnCode = vrep.simxSetJointTargetPosition(self.clientID, jointHandles[i], targetPositions[i], vrep.simx_opmode_blocking)
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
