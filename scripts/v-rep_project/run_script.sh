#/!/bin/sh
echo "Running evaluated_curriculum.py ..."
host=$(hostname)
python evaluate_curriculum.py </dev/null > "${host}_script_output.log" 2> "${host}_script_error.log"
echo "evaluated_curriculum.py ended."
