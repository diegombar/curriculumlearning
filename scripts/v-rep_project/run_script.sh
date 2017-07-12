#/!/bin/sh
echo "Running script..."
python evaluate_curriculum.py </dev/null >script_output.log 2>script_error.log
echo "evaluated_curriculum ended."