import os, sys, csv 
import json 
from collections import defaultdict

task_flag = False
task_code = "---"
json_data = defaultdict()

with open(sys.path[0] + "/results/results.txt", "r") as f:
    for idx, line in enumerate(f.readlines()):
        #print(idx)
        if line in ['\n', '\r\n']:
            continue

        if line.startswith("****"):
            task_flag = not task_flag
            continue

        if task_flag:
            task_code = line.strip()
            continue

        print(task_code)
        row = line.split("\t")

        model  = (row[0].strip()[:-4])
        sample = (row[1].strip())
        scores = (row[3:-1])

        json_data[model] = json_data.get(model, {})
        json_data[model][task_code] = json_data[model].get(task_code, {}) 
        json_data[model][task_code][sample] = scores

        
    print("Done Processing ... ")
    #print(json_data)
    with open(sys.path[0] + '/results/results.json', 'w') as js:
        json.dump(json_data, js)
        
                
