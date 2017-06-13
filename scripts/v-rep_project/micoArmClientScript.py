
# Lua implementation example
# vel=35
# accel=10
# jerk=5
# currentVel={0,0,0,0,0,0}
# currentAccel={0,0,0,0,0,0}
# maxVel={vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180}
# maxAccel={accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180}
# maxJerk={jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180}
# targetVel={0,0,0,0,0,0}

# targetPos1={90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180}
# simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos1,targetVel,{1,0,0,-1,-1,-1})

# targetPos2={90*math.pi/180,135*math.pi/180,225*math.pi/180,180*math.pi/180,180*math.pi/180,350*math.pi/180}
# simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos2,targetVel,{0,0,0,1,1,1})

# targetPos3={math.pi,math.pi,math.pi,math.pi,math.pi,math.pi}
# simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos3,targetVel,{-1,0,0,1,1,1})


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

print('Mico Arm Program started')
vrep.simxFinish(-1)# close all opened connections
# connect to V-Rep Remote Api Server
clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5) # Connect to V-REP

#Arm
pi = 3.1416
portNb = 19998
jointHandles = [0]*6

vel = 35
accel = 10
jerk = 5

currentVel = [0] * 6
currentAccel = [0] * 6
maxVel = [vel*pi/180] * 6
maxAccel = [accel*pi/180] * 6
maxJerk = [jerk*pi/180] * 6
targetVel = [0] * 6

targetPos1 = [90*pi/180] * 6
targetPos2 = [90*pi/180, 135*pi/180, 225*pi/180, 180*pi/180, 180*pi/180,350*pi/180]
targetPos3 = [pi] * 6
targetPos4 = [pi, 135*pi/180, 225*pi/180, 180*pi/180, 180*pi/180, 350*pi/180]

#Hand
closingVel = -0.04
# if (not closing) then
#     simSetJointTargetVelocity(j0,-closingVel)
#     simSetJointTargetVelocity(j1,-closingVel)
# else
#     simSetJointTargetVelocity(j0,closingVel)
#     simSetJointTargetVelocity(j1,closingVel)
# end

#testing setTargetVelocity
tVel = 2.0


# if len(sys.argv) >= 10:
#     portNb = int(sys.argv[1])
#     jointHandles = list(map(int,sys.argv[2:8]))
#     fingersH1 = int(sys.argv[8])
#     fingersH2 = int(sys.argv[9])
# else:
#     print("Indicate following arguments: 'portNumber jointHandles'")
#     time.sleep(5.0)
#     sys.exit(0)

def openHand(clientID):
    errorCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, -closingVel, vrep.simx_opmode_oneshot)
    errorCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, -closingVel, vrep.simx_opmode_oneshot)

def closeHand(clientID):
    errorCode = vrep.simxSetJointTargetVelocity(clientID, fingersH1, closingVel, vrep.simx_opmode_oneshot)
    errorCode = vrep.simxSetJointTargetVelocity(clientID, fingersH2, closingVel, vrep.simx_opmode_oneshot)

def setJointTargetPositions(clientID, targetPositions):
    for i in range(6):
        errorCode = vrep.simxSetJointTargetPosition(clientID, jointHandles[i], targetPositions[i], vrep.simx_opmode_oneshot)
        if errorCode != vrep.simx_return_ok:
            print("SetJointTargetPosition got error code: %s" % errorCode)


# clientID = vrep.simxStart('127.0.0.1', portNb, True, True, 2000, 5)

# Start simulation
vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
if clientID != -1:
    print('Connected to remote API server')
    jointHandles = [0] * 6
    # get Handles
    for i in range(0,6):
        returnCode, jointHandles[i] = vrep.simxGetObjectHandle(clientID, 'Mico_joint' + str(i+1), vrep.simx_opmode_blocking)
    if returnCode != vrep.simx_return_ok:
                print("GetHandle got error code: %s" % returnCode)
    # jointHandles[0] = vrep.simxGetObjectHandle(clientID, 'Mico_joint1', vrep.simx_opmode_blocking)
    # jointHandles[1] = vrep.simxGetObjectHandle(clientID, 'Mico_joint2', vrep.simx_opmode_blocking)
    # jointHandles[2] = vrep.simxGetObjectHandle(clientID, 'Mico_joint3', vrep.simx_opmode_blocking)
    # jointHandles[3] = vrep.simxGetObjectHandle(clientID, 'Mico_joint4', vrep.simx_opmode_blocking)
    # jointHandles[4] = vrep.simxGetObjectHandle(clientID, 'Mico_joint5', vrep.simx_opmode_blocking)
    # jointHandles[5] = vrep.simxGetObjectHandle(clientID, 'Mico_joint6', vrep.simx_opmode_blocking)
    returnCode, fingersH1 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor1', vrep.simx_opmode_blocking)
    returnCode, fingersH2 = vrep.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor2', vrep.simx_opmode_blocking)
    returnCode, mic = vrep.simxGetObjectHandle(clientID, 'Mico', vrep.simx_opmode_blocking)

    print('joints',jointHandles)
    print('f1',fingersH1)
    print('f2',fingersH1)
    print('micoRobot',mic)

    while vrep.simxGetConnectionId(clientID) != -1:
        # closeHand(clientID)
        # setJointTargetPositions(clientID, targetPos1)
        # errorCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[0], tVel, vrep.simx_opmode_blocking)
        # if errorCode != vrep.simx_return_ok:
        #     print("SetJointTargetVelocity got error code: %s" % errorCode)
        # time.sleep(2.0) #in sec

        # openHand(clientID)
        # setJointTargetPositions(clientID, targetPos2)
        # time.sleep(2.0) #in sec

        # closeHand(clientID)
        # setJointTargetPositions(clientID, targetPos3)
        # time.sleep(2.0) #in sec

        # testing settargetvel with arrow pressing
        c = readchar.readchar()
        if c == 'a':    
            return_code = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], tVel, vrep.simx_opmode_blocking)
            if errorCode != vrep.simx_return_ok:
                print("SetJointTargetPosition got error code: %s" % errorCode)
        elif c == 'd':
            errorCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], -tVel, vrep.simx_opmode_blocking)
            if errorCode != vrep.simx_return_ok:
                print("SetJointTargetPosition got error code: %s" % errorCode)
        elif c == 's':
            errorCode = vrep.simxSetJointTargetVelocity(clientID, jointHandles[5], 0, vrep.simx_opmode_blocking)
            if errorCode != vrep.simx_return_ok:
                print("SetJointTargetPosition got error code: %s" % errorCode)
        print('char=',c)       # --> gives me "0"
        time.sleep(0.1)

    vrep.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')

print('Mico Arm Program ended')


### send string signal to execute setTargetPositions in the server side

# #pack lists of int lists into a single string
# def pack2DIntArray(intLists):
#     stringData = ""
#     for intList in intLists:
#         stringData += vrep.simxPackInts(intList)
#     return stringData

# #pack lists of float lists into a single string
# def pack2DFloatArray(floatLists):
#     stringData = ""
#     for floatList in floatLists:
#         stringData += vrep.simxPackInts(floatList)
#     return stringData

# intLists = jointHandles
# floatLists = [currentVel, currentAccel, maxVel, maxAccel, maxJerk, targetPos, targetVel]
# stringData = pack2DIntArray(intLists)+pack2DFloatArray(floatLists)
# vrep.simxSetStringSignal(clientID, "moveToPosition", stringData, vrep.simx_opmode_oneshot_wait)

# while True:
#     returnCode, signalValue = vrep.simxGetStringSignal(clientID, "moveToPosition", vrep.simx_opmode_oneshot_wait)

# explore simxCallScriptFunction to call v-rep Lua script functions?