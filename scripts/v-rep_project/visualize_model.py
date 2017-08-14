import training_independent_joints as training
import os.path
import time

timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()

current_dir_path = os.path.dirname(os.path.realpath(__file__))
experiments_dir_path = os.path.join(current_dir_path, "trained_models_and_results")
experiment_name = 'visualizing_algorithm_'
folder_name = experiment_name + '_' + timestr
vis_experiment_dir_path = os.path.join(experiments_dir_path, folder_name)

# tasks

TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

targetRelativePos = (0.0, 0.7) #relative x, y in metres

###### CHOOSE

task = TASK_REACH_CUBE
vel = 1


#model to load

# model1_ep400 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","saved_checkpoints","checkpoint_model-400")

# model1_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","trained_model","final_model-3000") #success

# model2_ep400 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-03_15-24-03-success","saved_checkpoints","checkpoint_model-400")


# model2_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-11_14-19-11","trained_model","final_model-3000")

# model3_ep900 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-12_01-32-42","saved_checkpoints","checkpoint_model-900")


# model3_ep3000 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-12_01-32-42","trained_model","final_model-3000")

# new_model = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "model_and_results_2017-Jul-26_09-41-08","trained_model","final_model-400")

# vel2 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_17-00-30_vel=2","trained_model","final_model-400")
# vel1= os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_20-30-32_vel=1","trained_model","final_model-400")
# vel05 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-26_23-38-21_vel=05","trained_model","final_model-400")
# vel025 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "decreasing_speed","model_and_results_2017-Jul-27_02-49-34_vel=025","trained_model","final_model-400")



# model_1600ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-22_21-57-58_1600ep/trained_model",
#    "final_model-1600") #good results with e_min=0.1

# model_1600ep_vel1_1 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-25_02-48-01_1600ep/trained_model",
#    "final_model-1600")

# model_2800ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-24_07-19-16_2800ep/trained_model",
#    "final_model-2800")

# model_2000ep_vel1_0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "Testing different num of episodes/model_and_results_2017-Jul-25_12-02-10_2000ep/trained_model",
#    "final_model-2000") #good results with e_min=0.1


# model_800ep_vel1_emin0 = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "e_min=0, different num_ep, steps=200, vel=1_2017-Jul-26_22-22-21/model_and_results_2017-Jul-27_02-26-31/trained_model",
#    "final_model-800")

# reach_then_push = os.path.join(
#    current_dir_path,"trained_models_and_results",
#    "CL_reach_then_push_2017-Jul-26_20-27-48/model_and_results_2017-Jul-27_07-30-58",
#    "trained_model",
#    "final_model-400")

# model_longer_training_025 = os.path.join(experiments_dir_path,
#    "longer_training_vel025_2017-Jul-27_17-29-43/model_and_results_2017-Jul-27_17-29-43/trained_model",
#    "final_model-1000")

cl_decreasing_speeds = os.path.join(experiments_dir_path,
   "cl_decreasing_speeds_2017-Aug-10_21-22-07/model_and_results_2017-Aug-10_21-22-07/trained_model",
   "final_model-1666")

no_cl_5000_vel025 = os.path.join(experiments_dir_path,
   "no_curriculum_vel_025_2017-Aug-12_19-01-58/model_and_results_2017-Aug-12_19-01-58/trained_model",
   "final_model-5000") #3 hlayers x 100 neurons

no_cl_5000_vel1 = os.path.join(experiments_dir_path,
   "no_curriculum_vel_1_2017-Aug-10_22-39-08/model_and_results_2017-Aug-10_22-39-08/trained_model",
   "final_model-5000") #2 hlayers x 50 neurons


model_to_load = no_cl_5000_vel1

#load model

trainDQL_args = dict(
                    experiment_dir_path=vis_experiment_dir_path,
                    num_hidden_layers=2,
                    num_neurons_per_hidden=50,
                    num_episodes=10,  #400
                    max_steps_per_episode=200,  #200
                    e_min=0.01,
                    task=task,
                    model_saving_period=100,
                    lrate=1E-3,  # 1E-3 seems to work fine
                    batch_size=32,
                    replay_start_size=50000,
                    replay_memory_size=500000,
                    showGUI=True,
                    velocity=vel,  # 1.0 seems to work fine
                    model_to_load_file_path=model_to_load,
                    use_variable_names=True,
                    skip_training=True,
                    notes=experiment_name,
                    previous_norm=False,
                    targetRelativePos=targetRelativePos,
                    nSJoints=6,
                    nAJoints=6,
                    portNb=1997
                    )

training.trainDQL(**trainDQL_args)
#note: use previous_norm for first few models (angles were normalized to [0,2] (now fixed))