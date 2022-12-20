import sys 
import json
from turtle import color
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
import colorsys

def get_color(red_to_green):
    assert 0 <= red_to_green <= 1
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]

    minHue = 120/360 #40/360 <--- for bert
    maxHue = 120/360 #140/360 <--- for bert
    hue = red_to_green*maxHue + (1-red_to_green)*minHue;    

    #120 deg = 1/3  360 deg = 0.333  = 1 
    # 60 deg = 1/6  360 deg = 0.166  = 0
    r, g, b = colorsys.hsv_to_rgb(hue, red_to_green, 0.9)
    return list(map(lambda x: int(255 * x), (r, g, b)))

def plot_heatmap_results_only_for_best_layer_score(json_data, models, task_codes, samples):
    #for task_code in task_codes: 
        for sample in samples:
            #### gather model scores across a sample for ONE task at a time
            plot_data = []
            for model in models:
                model_plt_data = []
                for task_code in task_codes:
                    scores = json_data[model][task_code][sample]
                    scores = [float(item) for item in scores]

                    score_item = max(scores)
                    #score_item = int((score_item - (scores[0])) / (100 - (scores[0])) * 100) #NOTE: Uncomment to get normalized scores

                    model_plt_data.append(score_item)
                plot_data.append(model_plt_data)

            #### plot the dict
            df = pd.DataFrame(plot_data, columns=list(range(len(task_codes))), index=models)
            save_path = sys.path[0] + '/FIGS/HEATMAPS/' + 'table_results_' + '_BEST_LAYER_' + '_' + sample +'_samples__models_annot_mini_NORMALIZED.png'
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(12, 5), dpi=600)
            cm = sns.color_palette('Greens', as_cmap=True)
            ax = sns.heatmap(df, cmap=cm, linewidths=0, annot=True, yticklabels=models, xticklabels=task_codes, linecolor="white", vmin=0, vmax=100, cbar=False, square=20, fmt="3.0f") # annot=True shows the percentage values 
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=7)
            ax.xaxis.set_ticks_position('top')
        
            ax.set_facecolor('#ebfbe6') # add 80 to change alpha channel
            ax.figure.subplots_adjust(left = 0.3) 
            ax.figure.savefig((save_path), dpi=600)      
            plt.close()     

def plot_heatmap_results(json_data, models, task_codes, samples):
    for task_code in task_codes: 
        for sample in samples:
            #### gather model scores across a sample for ONE task at a time
            plot_data = []
            for model in models:
                scores = json_data[model][task_code][sample]
                plot_data.append([float(item) for item in scores])

            maxbin = 0
            #### make latex table plots
            for i, plot_data_list in enumerate(plot_data):
                print(models[i], end=" & ")

                for item in plot_data_list:
                    #print(" & ".join(plot_data_list), end="\\ \n")

                    task_scores_adj = item
                    task_scores_max = max(plot_data_list) + 5
                    task_scores_min = plot_data_list[0] - 5
                    normalized_value = float(float(task_scores_adj) - float(task_scores_min))/(float(task_scores_max) - float(task_scores_min))  

                    if task_scores_max >= maxbin:
                        maxbin = task_scores_max                                  

                    color_map = get_color(normalized_value)
                    color_map = [str(cm) for cm in color_map]
                    print(" {\cellcolor[RGB]{"+color_map[0]+","+color_map[1]+","+color_map[2]+"} "+ str(task_scores_adj) + " } ", end=" & ")
                print("\\\\")


            #### plot the dict
            df = pd.DataFrame(plot_data, columns=list(range(13)), index=models)
            save_path = sys.path[0] + '/results/HEATMAPS/' + task_code + '/' + 'table_results_' + task_code + '_' + sample +'_models_annot_mini.png'
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 5), dpi=600)
            cm = sns.color_palette('Greens', as_cmap=True)
            ax = sns.heatmap(df, cmap=cm, linewidths=0, yticklabels=models, linecolor="white", vmin=0, vmax=maxbin, xticklabels=False, cbar=False, square=10, fmt="3.0f") # annot=True shows the percentage values 
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
        
            ax.set_facecolor('#ebfbe6') # add 80 to change alpha channel
            ax.figure.subplots_adjust(left = 0.3) 
            ax.figure.savefig((save_path), dpi=600)      
            plt.close()     

def plot_graph_results(json_data, models, task_codes, samples):
    for task_code in task_codes: 
        for sample in samples:
            #### gather model scores across a sample for ONE task at a time
            plot_data = []
            for model in models:
                scores = json_data[model][task_code][sample]
                plot_data.append([float(item) for item in scores])   

            #### plot the dict
            df = pd.DataFrame(plot_data, columns=list(range(13)), index=models)        
            save_path = sys.path[0] + '/results/GRAPHS/' + task_code + '/' + 'table_results_' + task_code + '_' + sample +'_models.png'
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)                      

            plt.figure(figsize=(8, 5), dpi=600)
            #plt.grid(True, color='whitesmoke', which="major")
            sns.set(font="Verdana")

            ax = sns.lineplot(data=df.T, dashes=False, palette="flare")#"RdYlGn_r")#"gnuplot")#"Set1")#"CMRmap")#"RdYlGn_r")#"CMRmap")#"flare" #"YlOrRd_r")
            ax.set_xticks(range(13)) 
            ax.set_xticklabels(list(range(13)))
            ax.set_yticks(range(0, 110, 10)) 
            ax.set_yticklabels(list(range(0, 110, 10)))
            # ax.figure.subplots_adjust(left = 0.3) 
            ax.figure.savefig((save_path), dpi=600)  
            plt.close()                 



def main():
    results = sys.path[0] + "/results/results.json" # path to processed / combined JSON

    with open(results) as js:
        json_data = json.load(js)

    models     = [ # TODO: put models in models.config file
        "BERT", 
        "CodeBERT", 
        "CodeBERTa", 
        "CodeReviewer",
        "CodeT5", 
        "GraphCodeBERT", 
        "JavaBERT-KIEL", 
        "PLBART-mtjava", 
        "UnixCoder-unimodal", 
    ]

    samples    = ["100", "1k", "10k"]
    task_codes = ['KTX', 'IDN', 'LEN', 'TYP', 'REA', 'JBL', 'SRI', 'SRK', 'SCK', 'OCU', 'VCU', 'CSC', 'MXN', 'CPX', 'NPT'] # TODO: put tasks in tasks.config file
    
    plot_heatmap_results(json_data, models, task_codes, samples)
    #plot_heatmap_results_only_for_best_layer_score(json_data, models, task_codes, samples)
    #plot_graph_results(json_data, models, task_codes, samples)


if __name__ == '__main__':
    main()


