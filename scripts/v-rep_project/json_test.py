import json
import numpy as np

my_data = [list(range(5)) for i in range(5)]

print("sent: ",my_data)

with open('temp.json', "w") as file:
    json.dump(eval('my_data'), file)

with open('temp.json', "r") as file:
    my_saved_data = json.load(file)
    print("received:", my_saved_data)

#works
