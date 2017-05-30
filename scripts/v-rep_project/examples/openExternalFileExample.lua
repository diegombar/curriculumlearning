-- Get application and scene paths --
-- appPath = simGetStringParameter(sim_stringparam_application_path)
scenePath = simGetStringParameter(sim_stringparam_scene_path)
-- Using functions in external Lua files: --
-- package.path = package.path .. "D:path/To/File/?.lua;"
-- require("myfunctions")
-- data=foo(something)
-- data2=bar(somethingelse)

-- Running external Lua files: --
-- local f = loadfile('/home/diego/curriculumlearning/scripts/v-rep_project/bubbleRobServerScript.lua')
local f = loadfile(scenePath .. '/bubbleRobServerScript.lua')
return f()