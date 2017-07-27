import training_independent_joints as training
import os.path
import time

timestr = time.strftime("%Y-%b-%d_%H-%M-%S",time.gmtime()) #or time.localtime()

current_dir_path = os.path.dirname(os.path.realpath(__file__))


#tasks
TASK_REACH_CUBE = 1
TASK_PUSH_CUBE_TO_TARGET_POSITION = 2

targetRelativePos = (0.0, 0.7) #relative x, y in metres

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

model_1600ep_vel1_0 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "Testing different num of episodes/model_and_results_2017-Jul-22_21-57-58_1600ep/trained_model",
   "final_model-1600") #good results with e_min=0.1

model_1600ep_vel1_1 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "Testing different num of episodes/model_and_results_2017-Jul-25_02-48-01_1600ep/trained_model",
   "final_model-1600")

model_2800ep_vel1_0 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "Testing different num of episodes/model_and_results_2017-Jul-24_07-19-16_2800ep/trained_model",
   "final_model-2800")

model_2000ep_vel1_0 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "Testing different num of episodes/model_and_results_2017-Jul-25_12-02-10_2000ep/trained_model",
   "final_model-2000") #good results with e_min=0.1


model_800ep_vel1_emin0 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "e_min=0, different num_ep, steps=200, vel=1_2017-Jul-26_22-22-21/model_and_results_2017-Jul-27_02-26-31/trained_model",
   "final_model-800")

model_to_load = model_800ep_vel1_emin0


#load model
training.trainDQL(experiment_folder_name='visualizing_algorithm_'+timestr,
                  num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  num_episodes=10,
                  max_steps_per_episode=300,
                  e_min=0.01, #or 0.01
                  task=TASK_REACH_CUBE,
                  showGUI=True,
                  velocity=1.0,
                  model_to_load_file_path=model_to_load,
                  use_variable_names=True, #test changing
                  skip_training=True,
                  notes="visualizing loaded model",
                  previous_norm=False)

#note: use previous_norm for first few models (angles were normalized to [0,2] (now fixed))