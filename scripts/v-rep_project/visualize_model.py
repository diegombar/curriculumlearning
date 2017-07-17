import training_independent_joints as training
import os.path

current_dir_path = os.path.dirname(os.path.realpath(__file__))

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


model3_ep3000 = os.path.join(
   current_dir_path,"trained_models_and_results",
   "model_and_results_2017-Jul-12_01-32-42","trained_model","final_model-3000")

model_to_load = model3_ep3000


#load model
training.trainDQL(num_hidden_layers=2,
                  num_neurons_per_hidden=50,
                  num_episodes=10,
                  max_steps_per_episode=500,
                  e_min=0.0,
                  showGUI=True,
                  velocity=0.3,
                  model_to_load_file_path=model_to_load,
                  use_variable_names=True,
                  skip_training=True,
                  notes="visualizing loaded model")

#note: in first few models, angles were normalized to [0,2] (now fixed)