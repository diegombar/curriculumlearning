simSetThreadSwitchTiming(2)
-- Get scene path:
scenePath = simGetStringParameter(sim_stringparam_scene_path)

-- Get handles:
jointHandles={-1,-1,-1,-1,-1,-1}
for i=1,6,1 do
    jointHandles[i]=simGetObjectHandle('Mico_joint'..i)
end
fingersH1=simGetObjectHandle("MicoHand_fingers12_motor1")
fingersH2=simGetObjectHandle("MicoHand_fingers12_motor2")

-- Choose a port that is probably not used:
simSetThreadAutomaticSwitch(false)
local portNb=simGetInt32Parameter(sim_intparam_server_port_next)
local portStart=simGetInt32Parameter(sim_intparam_server_port_start)
local portRange=simGetInt32Parameter(sim_intparam_server_port_range)
local newPortNb=portNb+1
if (newPortNb>=portStart+portRange) then
    newPortNb=portStart
end
simSetInt32Parameter(sim_intparam_server_port_next,newPortNb)
simSetThreadAutomaticSwitch(true)

-- Check what OS we are using:
platf=simGetInt32Parameter(sim_intparam_platform)
if (platf==0) then
    pluginFile='v_repExtRemoteApi.dll'
end
if (platf==1) then
    pluginFile='libv_repExtRemoteApi.dylib'
end
if (platf==2) then
    pluginFile='libv_repExtRemoteApi.so'
end

-- Check if the required remote Api plugin is there:
moduleName=0
moduleVersion=0
index=0
pluginNotFound=true
while moduleName do
    moduleName,moduleVersion=simGetModuleName(index)
    if (moduleName=='RemoteApi') then
        pluginNotFound=false
    end
    index=index+1
end

if (pluginNotFound) then
    -- Plugin was not found
    simDisplayDialog('Error',"Remote Api plugin was not found. ('"..pluginFile.."')&&nSimulation will not run properly",sim_dlgstyle_ok,true,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
else
    -- Ok, we found the plugin.
    -- We first start the remote Api server service (this requires the v_repExtRemoteApi plugin):
    simExtRemoteApiStart(portNb) -- this server function will automatically close again at simulation end
    -- simExtRemoteApiStart(19998)
    -- Now we start the client application:
    command_path="/home/diego/anaconda3/bin/python"
    handles= table.concat(jointHandles, " ").." "..fingersH1.." "..fingersH2
    command_args=scenePath.."/micoArmClientScript.py "..portNb.." "..handles
    result = simLaunchExecutable(command_path, command_args, 1) -- set the last argument to 1 to see the console of the launched client
    if (result==-1) then
        -- The executable could not be launched!
        simDisplayDialog('Error',"'micoArmClientScript' could not be launched. &&nSimulation will not run properly",sim_dlgstyle_ok,true,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
    end
end

-- Example script without remote API --
-- Set-up some of the RML vectors:

-- vel=35
-- accel=10
-- jerk=5
-- currentVel={0,0,0,0,0,0}
-- currentAccel={0,0,0,0,0,0}
-- maxVel={vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180}
-- maxAccel={accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180}
-- maxJerk={jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180}
-- targetVel={0,0,0,0,0,0}

-- targetPos1={90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180}
-- simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos1,targetVel,{1,0,0,-1,-1,-1})

-- targetPos2={90*math.pi/180,135*math.pi/180,225*math.pi/180,180*math.pi/180,180*math.pi/180,350*math.pi/180}
-- simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos2,targetVel,{0,0,0,1,1,1})

-- targetPos3={math.pi,math.pi,math.pi,math.pi,math.pi,math.pi}
-- simRMLMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos3,targetVel,{-1,0,0,1,1,1})
