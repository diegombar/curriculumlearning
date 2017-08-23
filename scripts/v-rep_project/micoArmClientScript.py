#!/usr/bin/env python3
import os
import signal
import subprocess
# import sys
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

print('Mico Arm Program started')


def printlog(functionName, returnCode):
    if returnCode == vrep.simx_return_ok:
        print("{} successful".format(functionName))
    else:
        print("{} got error code: {}".format(functionName, returnCode))


def degrees2Radians(positionsInDeg):
    # use np.array
    return positionsInDeg * np.pi / 180


# launch v-rep
portNb = 19998
home_path = os.path.expanduser('~')
vrepPath = os.path.join(home_path, "V-REP_PRO_EDU_V3_4_0_Linux", "vrep.sh")
vrep_cmd = [vrepPath, '-gREMOTEAPISERVERSERVICE_' + str(portNb) + '_FALSE_FALSE']
vrepProcess = subprocess.Popen(vrep_cmd, shell=False, preexec_fn=os.setsid)

# connect to V-Rep Remote Api Server
vrep.simxFinish(-1)  # close all opened connections
clientID = vrep.simxStart('127.0.0.1', portNb, True, False, 5000, 5)  # Connect to V-REP

if clientID == -1:
    print('Failed connecting to remote API server')
else:
    print('Connected to remote API server')
    # load scene
    time.sleep(5)  # to avoid errors
    scenePath = 'MicoRobot_no_gravity.ttt'
    returnCode = vrep.simxLoadScene(clientID, scenePath, 1, vrep.simx_opmode_oneshot_wait)  # vrep.simx_opmode_blocking is recommended
    printlog('simxLoadScene', returnCode)

    # when using child script
    # if len(sys.argv) >= 10:
    #     portNb = int(sys.argv[1])
    #     jointHandles = list(map(int,sys.argv[2:8]))
    #     fingersH1 = int(sys.argv[8])
    #     fingersH2 = int(sys.argv[9])
    # else:
    #     print("Indicate following arguments: 'portNumber jointHandles'")
    #     time.sleep(5.0)
    #     sys.exit(0)

    # get Handles
    jointHandles = [0] * 6
    for i in range(0, 6):
        returnCode, jointHandles[i] = vrep.simxGetObjectHandle(clientID, 'Mico_joint' + str(i + 1), vrep.simx_opmode_blocking)
        if i == 0:
            printlog('simxGetObjectHandle', returnCode)
    returnCode, fingersH1 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
    returnCode, fingersH2 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
    returnCode, jointsCollectionHandle = vrep.simxGetCollectionHandle(clientID, "sixJoints#", vrep.simx_opmode_blocking)
    returnCode, distToGoalHandle = vrep.simxGetDistanceHandle(clientID, "distanceToGoal#", vrep.simx_opmode_blocking)
    returnCode, distanceToGoal = vrep.simxReadDistance(clientID, distToGoalHandle, vrep.simx_opmode_streaming)  # start streaming

    print('jointHandles', jointHandles)
    # Arm
    pi = np.pi
    vel = 0.25  # looks fast in simulation!

    # some test target positions for the 6 joints, in radians
    targetPos0 = np.zeros(6)
    targetPosPi = np.array([np.pi] * 6)
    targetPosPiO2 = np.array([np.pi / 2] * 6)
    targetPos1 = np.array([np.pi / 2] * 6)
    targetPos2 = np.array([90, 135, 225, 180, 180, 350])
    targetPos2 = degrees2Radians(targetPos2)
    targetPos4 = np.array([180, 135, 225, 180, 180, 350])
    targetPos4 = degrees2Radians(targetPos4)

    targetPosInitial = np.array([1.0] * 6) * np.pi
    targetPosStraight = np.array([0.66, 1.0, 1.25, 1.5, 1.0, 1.0]) * np.pi
    targetPosHalfWayCube = np.array([0.66, 1.25, 1.25, 1.5, 1.0, 1.0]) * np.pi
    targetPosNearCube = np.array([0.66, 1.5, 1.25, 1.5, 1.0, 1.0]) * np.pi

    # checkGoal
    goalReached = False
    minDistance = 0.05  # one cm from goal

    # Hand
    # def openHand(clientID):
    #     closingVel = -0.04
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, -closingVel, vrep.simx_opmode_oneshot)
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, -closingVel, vrep.simx_opmode_oneshot)

    # def closeHand(clientID):
    #     closingVel = -0.04
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, closingVel, vrep.simx_opmode_oneshot)
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, closingVel, vrep.simx_opmode_oneshot)

    # set joints target positions
    def setJointTargetPositions(targetPositions):
        for i in range(6):
            returnCode = vrep.simxSetJointTargetPosition(clientID, jointHandles[i], targetPositions[i], vrep.simx_opmode_oneshot)
            if returnCode != vrep.simx_return_ok:
                print("SetJointTargetPosition got error code: %s" % returnCode)

    def enableControlLoop():
        for i in range(6):
            vrep.simxSetObjectIntParameter(clientID, jointHandles[i], vrep.sim_jointintparam_ctrl_enabled, 1, vrep.simx_opmode_blocking)

    def disableControlLoop():
        for i in range(6):
            vrep.simxSetObjectIntParameter(clientID, jointHandles[i], vrep.sim_jointintparam_ctrl_enabled, 0, vrep.simx_opmode_blocking)

    # Start simulation
    returnCode = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    printlog('simxStartSimulation', returnCode)

    # check server state before loop
    while vrep.simxGetConnectionId(clientID) != -1:
        # execute actions and get state here
        # testing settargetvel with arrow pressing
        c = readchar.readchar()
        print('char=', c)
        if c == 'a':
            # returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], vel, vrep.simx_opmode_blocking)
            # printlog('simxSetJointTargetVelocity', returnCode)
            setJointTargetPositions(targetPosInitial)
        elif c == 'd':
            # returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], -vel, vrep.simx_opmode_blocking)
            # printlog('simxSetJointTargetVelocity', returnCode)
            setJointTargetPositions(targetPosHalfWayCube)
        elif c == 's':
            # returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], 0, vrep.simx_opmode_blocking)
            # printlog('simxSetJointTargetVelocity', returnCode)
            setJointTargetPositions(targetPosStraight)
        elif c == 'w':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[4], vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'r':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[4], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'e':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[4], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'f':
            # returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[3], vel, vrep.simx_opmode_blocking)
            # printlog('simxSetJointTargetVelocity', returnCode)
            setJointTargetPositions(targetPosNearCube)
        elif c == 'h':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[3], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'g':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[3], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 't':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[2], vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'u':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[2], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'y':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[2], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'j':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[1], vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'l':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[1], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'k':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[1], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'i':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[0], vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'p':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[0], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'o':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[0], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'z':
            # stop simulation
            returnCode = vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
            printlog('simxStopSimulation', returnCode)
            running = True
            while running:
                returnCode, ping = vrep.simxGetPingTime(clientID)
                returnCode, serverState = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
                running = serverState
            # Start simulation
            returnCode = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
            printlog('simxStartSimulation', returnCode)
        elif c == 'x':
            # testing getState
            returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(clientID, jointsCollectionHandle, 15, vrep.simx_opmode_streaming)  # or simx_opmode_blocking (not recommended)
            jointPositions = np.array(floatData[0::2])  # take elements at odd positions (even corresponds to torques)
            jointPositions = jointPositions % (2 * np.pi)  # take values in [0, 2*pi]
            print('jointPositions: ', jointPositions)
        elif c == 'c':
            enableControlLoop()
            print('Joints control loop enabled.')
        elif c == 'v':
            disableControlLoop()
            print('Joints control loop disabled.')
        elif c == 'q':
            break

        # testing getState
        returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(clientID, jointsCollectionHandle, 15, vrep.simx_opmode_streaming)  # or simx_opmode_blocking (not recommended)
        jointPositions = np.array(floatData[0::2])  # take elements at odd positions (even corresponds to torques)
        jointPositions = jointPositions % (2 * np.pi)  # take values in [0, 2*pi]
        print('jointPositions: ', jointPositions)

        # get reward (proximity sensor)
        returnCode, distanceToGoal = vrep.simxReadDistance(clientID, distToGoalHandle, vrep.simx_opmode_buffer)
        print("distance to goal:", distanceToGoal)
        # reward =

        # checkGoalReached
        if distanceToGoal < minDistance:
            goalReached = True
        time.sleep(0.1)

        # end of execution loop

    # stop simulation
    returnCode = vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    printlog('simxStopSimulation', returnCode)

    running = True
    while running:
        returnCode, ping = vrep.simxGetPingTime(clientID)
        returnCode, serverState = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
        running = serverState

    returnCode = vrep.simxCloseScene(clientID, vrep.simx_opmode_blocking)
    printlog('simxCloseScene', returnCode)

    vrep.simxFinish(clientID)

# close v-rep
# vrepProcess.terminate()
# vrepProcess.kill()
os.killpg(os.getpgid(vrepProcess.pid), signal.SIGTERM)

print('Mico Arm Program ended')

