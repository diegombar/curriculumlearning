import tensorflow as tf
import os.path


current_dir_path = os.path.dirname(os.path.realpath(__file__)) 
successful_model = os.path.join(
   current_dir_path,"trained_models_and_results",
   "model_and_results_2017-Jul-03_15-24-03-success","trained_model","final_model-3000")

new_model = os.path.join(
   current_dir_path,"trained_models_and_results",
   "model_and_results_2017-Jul-26_09-41-08","trained_model","final_model-400")

variables = tf.contrib.framework.list_variables(new_model)

for variable in variables:
	print(variable)