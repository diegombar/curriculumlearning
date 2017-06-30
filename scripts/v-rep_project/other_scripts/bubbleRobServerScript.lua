simSetThreadSwitchTiming(2)

-- Get application and scene paths --
-- appPath = simGetStringParameter(sim_stringparam_application_path)
scenePath = simGetStringParameter(sim_stringparam_scene_path)

-- Get some handles first:
local leftMotor=simGetObjectHandle("remoteApiControlledBubbleRobLeftMotor") -- Handle of the left motor
local rightMotor=simGetObjectHandle("remoteApiControlledBubbleRobRightMotor") -- Handle of the right motor
local noseSensor=simGetObjectHandle("remoteApiControlledBubbleRobSensingNose") -- Handle of the proximity sensor

-- Add a banner:
black={0,0,0,0,0,0,0,0,0,0,0,0}
purple={0,0,0,0,0,0,0,0,0,1,0,1}
simAddBanner("I am controlled via the Remote Api! ('bubbleRobClient' controlls me)",0,sim_banner_bitmapfont+sim_banner_overlay,nil,simGetObjectAssociatedWithScript(sim_handle_self),black,purple)

-- Choose a port that is probably not used (try to always use a similar code):
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
    -- result=simLaunchExecutable('bubbleRobClient',portNb.." "..leftMotor.." "..rightMotor.." "..noseSensor,1) -- set the last argument to 1 to see the console of the launched client
    command_path="/home/diego/anaconda3/bin/python"
    command_args=scenePath .. "/micoArmClientScript.py "..portNb.." "..leftMotor.." "..rightMotor.." "..noseSensor
    result = simLaunchExecutable(command_path, command_args, 1)
    if (result==-1) then
        -- The executable could not be launched!
        simDisplayDialog('Error',"'bubbleRobClient' could not be launched. &&nSimulation will not run properly",sim_dlgstyle_ok,true,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
    end
end

-- This thread ends here. The bubbleRob will however still be controlled by
-- the client application via the remote Api mechanism!

-- number result=simExtRemoteApiStop(number portNumber)
