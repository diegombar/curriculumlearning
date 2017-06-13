#!/usr/bin/env python3
print('Mico Arm Program started')

# launch v-rep
import os
import signal
import subprocess
vrepPath = "/home/diego/V-REP_PRO_EDU_V3_4_0_Linux/vrep.sh"
vrepProcess = subprocess.Popen(vrepPath, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid)

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
import sys
import time
import readchar
import numpy as np

def printlog(functionName, returnCode):
    if returnCode == vrep.simx_return_ok:
        print("{} successful".format(functionName))
    else:
        print("{} got error code: {}".format(functionName, returnCode))

# connect to V-Rep Remote Api Server
vrep.simxFinish(-1)# close all opened connections
portNb = 19998 # must match the portNb on server side specified in remoteApiConnections.txt
clientID=vrep.simxStart('127.0.0.1',portNb,True,False,5000,5) # Connect to V-REP

if clientID ==-1:
    print('Failed connecting to remote API server')
else:
    print('Connected to remote API server')
    # load scene
    # time.sleep(5) # to avoid errors
    scenePath = 'MicoRobot.ttt'
    returnCode = vrep.simxLoadScene(clientID, scenePath, 1, vrep.simx_opmode_oneshot_wait) # vrep.simx_opmode_blocking is recommended
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
    for i in range(0,6):
        returnCode, jointHandles[i] = vrep.simxGetObjectHandle(clientID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
        if i==0: printlog('simxGetObjectHandle', returnCode)
    returnCode, fingersH1 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
    returnCode, fingersH2 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
    returnCode, jointsCollectionHandle = vrep.simxGetCollectionHandle(clientID, "sixJoints#", vrep.simx_opmode_blocking)

    #Arm
    pi = np.pi
    vel = 5 # looks fast in simulation!

    #some test target positions for the 6 joints
    targetPos1 = [90*pi/180] * 6
    targetPos2 = [90*pi/180, 135*pi/180, 225*pi/180, 180*pi/180, 180*pi/180,350*pi/180]
    targetPos3 = [pi] * 6
    targetPos4 = [pi, 135*pi/180, 225*pi/180, 180*pi/180, 180*pi/180, 350*pi/180]

    #Hand
    # def openHand(clientID):
    #     closingVel = -0.04
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, -closingVel, vrep.simx_opmode_oneshot)
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, -closingVel, vrep.simx_opmode_oneshot)

    # def closeHand(clientID):
    #     closingVel = -0.04
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, closingVel, vrep.simx_opmode_oneshot)
    #     returnCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, closingVel, vrep.simx_opmode_oneshot)

    #set joints target positions
    # def setJointTargetPositions(clientID, targetPositions):
    #     for i in range(6):
    #         returnCode = vrep.simxSetJointTargetPosition(clientID, jointHandles[i], targetPositions[i], vrep.simx_opmode_oneshot)
    #         if returnCode != vrep.simx_return_ok:
    #             print("SetJointTargetPosition got error code: %s" % returnCode)

    # Start simulation
    returnCode = vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    printlog('simxStartSimulation', returnCode)
    
    # check server state before loop
    while vrep.simxGetConnectionId(clientID) != -1:
        #execute actions and get state here



        # testing settargetvel with arrow pressing
        c = readchar.readchar()
        print('char=',c)
        if c == 'a':    
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'd':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], -vel, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 's':
            returnCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], 0, vrep.simx_opmode_blocking)
            printlog('simxSetJointTargetVelocity', returnCode)
        elif c == 'q':
            break
        

        #testing getState
        returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(clientID, jointsCollectionHandle, 15, vrep.simx_opmode_streaming) # or simx_opmode_blocking (not recommended)
        jointPositions = np.array(floatData[0::2]) #take elements at odd positions (even corresponds to torques)
        jointPositions = jointPositions % (2 * np.pi) #take values in [0, 2*pi]

        #get reward (proximity sensor)
        returnCode, _, _, floatData, _ = vrep.simxGetObjectGroupData(clientID, jointsCollectionHandle, 13, vrep.simx_opmode_streaming)



        time.sleep(1)

        #end of execution loop

    # stop simulation
    returnCode = vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    printlog('simxStopSimulation', returnCode)

    running = True
    while running:
        returnCode, ping = vrep.simxGetPingTime(clientID)
        returnCode, serverState = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
        running = serverState

    returnCode = vrep.simxCloseScene(clientID,vrep.simx_opmode_blocking)
    printlog('simxCloseScene', returnCode)

    vrep.simxFinish(clientID)


#close v-rep
# vrepProcess.terminate()
# vrepProcess.kill()
os.killpg(os.getpgid(vrepProcess.pid), signal.SIGTERM)

print('Mico Arm Program ended')