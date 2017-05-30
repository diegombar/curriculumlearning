-- See the end of the script for instructions on how to do efficient grasping

if (sim_call_type==sim_childscriptcall_initialization) then 
    modelHandle=simGetObjectAssociatedWithScript(sim_handle_self)
    j0=simGetObjectHandle("MicoHand_fingers12_motor1")
    j1=simGetObjectHandle("MicoHand_fingers12_motor2")
    ui=simGetUIHandle('MicoHand')
    simSetUIButtonLabel(ui,0,simGetObjectName(modelHandle))
    closingVel=-0.04
end 

if (sim_call_type==sim_childscriptcall_cleanup) then 
 
end 

if (sim_call_type==sim_childscriptcall_actuation) then 
    closing=simBoolAnd16(simGetUIButtonProperty(ui,20),sim_buttonproperty_isdown)~=0
    
    if (not closing) then
        simSetJointTargetVelocity(j0,-closingVel)
        simSetJointTargetVelocity(j1,-closingVel)
    else
        simSetJointTargetVelocity(j0,closingVel)
        simSetJointTargetVelocity(j1,closingVel)
    end
    
    -- You have basically 2 alternatives to grasp an object:
    --
    -- 1. You try to grasp it in a realistic way. This is quite delicate and sometimes requires
    --    to carefully adjust several parameters (e.g. motor forces/torques/velocities, friction
    --    coefficients, object masses and inertias)
    --
    -- 2. You fake the grasping by attaching the object to the gripper via a connector. This is
    --    much easier and offers very stable results.
    --
    -- Alternative 2 is explained hereafter:
    --
    --
    -- a) In the initialization phase, retrieve some handles:
    -- 
    -- connector=simGetObjectHandle('MicoHand_attachPoint')
    -- objectSensor=simGetObjectHandle('MicoHand_attachProxSensor')
    
    -- b) Before closing the gripper, check which dynamically non-static and respondable object is
    --    in-between the fingers. Then attach the object to the gripper:
    --
    -- index=0
    -- while true do
    --     shape=simGetObjects(index,sim_object_shape_type)
    --     if (shape==-1) then
    --         break
    --     end
    --     if (simGetObjectInt32Parameter(shape,sim_shapeintparam_static)==0) and (simGetObjectInt32Parameter(shape,sim_shapeintparam_respondable)~=0) and (simCheckProximitySensor(objectSensor,shape)==1) then
    --         -- Ok, we found a non-static respondable shape that was detected
    --         attachedShape=shape
    --         -- Do the connection:
    --         simSetObjectParent(attachedShape,connector,true)
    --         break
    --     end
    --     index=index+1
    -- end
    
    -- c) And just before opening the gripper again, detach the previously attached shape:
    --
    -- simSetObjectParent(attachedShape,-1,true)
end 
