import pandas as pd
import numpy as np
from pathlib import Path
import json
from shapely.geometry import box
from fiona.crs import from_epsg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_boxplots(csv_file, save='False'):
    data = pd.read_csv(csv_file)
    # data.plot(kind='box')
    plt.boxplot(x=data, labels=list(data.columns.values))
    plt.ylabel(csv_file[:-4])
    plt.xticks(fontsize=8, rotation=20, ha='right')
    if save == 'True':
                plt.tight_layout()
                plt.savefig(f'{csv_file[:-4]}.png')
    plt.show()

def error_bars(csv_file, save='True'):
    data = pd.read_csv(csv_file)
    means = []
    std = []
    for (columnName, columnData) in data.iteritems():
         means.append(np.mean(columnData))
         std.append(np.std(columnData))
         
    labels=list(data.columns.values)
    # x_pos = np.arange(len(labels))
    x_pos = [0,0.8,1.6,2.4,3.2,4,4.8,5.6, 7,7.8,8.6, 10,10.8,11.6]

    fig, ax = plt.subplots()
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    fig.set_size_inches(10,4)

    #get current axes
    ax = plt.gca()
    #hide y-axis
    # ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    width = 0.5
    
    plt.xlim(-0.5, len(means) - 1 + 0.5)
    bar_container = ax.bar(x_pos, means,
        yerr=std, capsize=5, ecolor='grey', width = width, color =['#C3B898','#C3B898','#C3B898',
                                                                   '#C3B898','#C3B898','#C3B898',
                                                                   '#C3B898','#C3B898','#b35823','#b35823','#b35823',
                                                                   '#78B394', '#78B394', '#78B394'] )
    ax.bar_label(bar_container, color = 'white',  fmt='%.2f', fontsize=8, label_type='center')
    # Hide X and Y axes label marks
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_yticks([])

    # Save the figure and show
    # plt.ylabel(csv_file[:-4])
    plt.xticks(fontsize=8, rotation=20, ha='right')
    
    if save == 'True':
                # plt.rcParams['figure.figsize'] = [3,1]
                plt.tight_layout()
                plt.savefig(f'{csv_file[:-4]}_error.png')
    plt.show()

if __name__ == "__main__":
    
    plot_boxplots('ALL_loss_box.csv', save='True')
    error_bars('ALL_IoU_bonus.csv')