-- This is a threaded script, and is just an example!

jointHandles={-1,-1,-1,-1,-1,-1}
for i=1,6,1 do
    jointHandles[i]=simGetObjectHandle('Mico_joint'..i)
end

-- Set-up some of the RML vectors:
vel=35
accel=10
jerk=5
currentVel={0,0,0,0,0,0}
currentAccel={0,0,0,0,0,0}
maxVel={vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180}
maxAccel={accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180}
maxJerk={jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180}
targetVel={0,0,0,0,0,0}

targetPos1={90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180}
simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos1,targetVel,{1,0,0,-1,-1,-1})

targetPos2={90*math.pi/180,135*math.pi/180,225*math.pi/180,180*math.pi/180,180*math.pi/180,350*math.pi/180}
simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos2,targetVel,{0,0,0,1,1,1})

targetPos3={math.pi,math.pi,math.pi,math.pi,math.pi,math.pi}
simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos3,targetVel,{-1,0,0,1,1,1})
