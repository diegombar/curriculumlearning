import vrep
import sys
import time

print('BubbleRob Program started')

portNb = 0
leftMotorHandle = 0
rightMotorHandle = 0
sensorHandle = 0

if len(sys.argv) >= 5:
    portNb = int(sys.argv[1])
    leftMotorHandle = int(sys.argv[2])
    rightMotorHandle = int(sys.argv[3])
    sensorHandle = int(sys.argv[4])
else:
    print("Indicate following arguments: 'portNumber leftMotorHandle rightMotorHandle sensorHandle'")
    time.sleep(5000.0 / 1000.0)
    sys.exit(0)

vrep.simxFinish(-1) # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', portNb, True, True, 2000, 5)
if clientID != -1:
    print('Connected to remote API server')
    driveBackStartTime = -99000
    motorSpeeds = [0, 0]

    while vrep.simxGetConnectionId(clientID) != -1:
        (errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(clientID, sensorHandle, vrep.simx_opmode_streaming)
        if errorCode == vrep.simx_return_ok:
            simulationTime = vrep.simxGetLastCmdTime(clientID)
            if simulationTime - driveBackStartTime < 3000:
                motorSpeeds[0] = -3.1415 * 0.5
                motorSpeeds[1] = -3.1415 * 0.25
            else:
                motorSpeeds[0] = 3.1415
                motorSpeeds[1] = 3.1415
                if detectionState:
                    driveBackStartTime = simulationTime

            errorCode = vrep.simxSetJointTargetVelocity(clientID, leftMotorHandle, motorSpeeds[0], vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, rightMotorHandle, motorSpeeds[1], vrep.simx_opmode_oneshot)
            if errorCode != vrep.simx_return_ok:
                print("SetJointTargetVelocity got error code: %s" % errorCode)

        else:
            print("ReadProximitySensor got error code: %s" % errorCode)

        time.sleep(5.0 / 1000.0)

    vrep.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')