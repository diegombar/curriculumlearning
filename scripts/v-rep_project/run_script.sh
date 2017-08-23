#/!/bin/sh
echo "Running evaluated_curriculum.py ..."
host=$(hostname)
python comparison_plot.py </dev/null > "${host}_script_output.log" 2> "${host}_script_error.log"
echo "evaluated_curriculum.py ended."
