import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


#    _                      _    _            _                         
#   | |                    | |  | |          | |                        
#   | |     ___  ___ ___   | |__| | ___  __ _| |_ _ __ ___   __ _ _ __  
#   | |    / _ \/ __/ __|  |  __  |/ _ \/ _` | __| '_ ` _ \ / _` | '_ \ 
#   | |___| (_) \__ \__ \  | |  | |  __/ (_| | |_| | | | | | (_| | |_) |
#   |______\___/|___/___/  |_|  |_|\___|\__,_|\__|_| |_| |_|\__,_| .__/ 
#                                                                | |    


def generate_heatmap():
    
    for tool_length in [20, 30, 50]:
        
        df_losses = pd.read_csv(f'../results/neutouch_singletool_{tool_length}.csv',
                                names=['taxel', 'test_loss_mean', 'test_loss_std'],
                                dtype={'taxel': int, 'test_loss_mean': float, 'test_loss_std': float})

        left_pattern = np.array([[36, 37, 38, 39, 40],
                                 [31, 32, 33, 34, 35],
                                 [26, 27, 28, 29, 30],
                                 [21, 22, 23, 24, 25],
                                 [5 , 4 , 3 , 2 , 1 ],
                                 [10, 9 , 8 , 7 , 6 ],
                                 [15, 14, 13, 12, 11],
                                 [20, 19, 18, 17, 16]])

        right_pattern = np.array([[56, 57, 58, 59, 60],
                                  [51, 52, 53, 54, 55],
                                  [46, 47, 48, 49, 50],
                                  [41, 42, 43, 44, 45],
                                  [65, 64, 63, 62, 61],
                                  [70, 69, 68, 67, 66],
                                  [75, 74, 73, 72, 71],
                                  [80, 79, 78, 77, 76]])

        left_distribution = np.zeros_like(left_pattern, dtype=float)
        right_distribution = np.zeros_like(right_pattern, dtype=float)

        for _, row in df_losses.iterrows():

            if row.taxel <= 40: left_distribution[left_pattern == row.taxel] = row.test_loss_mean
            else: right_distribution[right_pattern == row.taxel] = row.test_loss_mean

        fig, axs = plt.subplots(1, 2, figsize=(6, 6))

        axs[0].axis('off')
        axs[1].axis('off')

        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        vmin = df_losses.test_loss_mean.min()
        vmax = df_losses.test_loss_mean.max()

        axs[0].imshow(left_distribution, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].imshow(right_distribution, cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(8):
            for j in range(5):
                axs[0].text(j, i, '({:02d})\n{:0.4f}'.format(left_pattern[i, j], left_distribution[i, j]), ha='center', va='center', color='k')
                axs[1].text(j, i, '({:02d})\n{:0.4f}'.format(right_pattern[i, j], right_distribution[i, j]), ha='center', va='center', color='k')

        fig.savefig(f'neutouch_singletool_{tool_length}_heatmap.pdf', bbox_inches='tight')


#    _                      _    _ _     _                                  
#   | |                    | |  | (_)   | |                                 
#   | |     ___  ___ ___   | |__| |_ ___| |_ ___   __ _ _ __ __ _ _ __ ___  
#   | |    / _ \/ __/ __|  |  __  | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \ 
#   | |___| (_) \__ \__ \  | |  | | \__ \ || (_) | (_| | | | (_| | | | | | |
#   |______\___/|___/___/  |_|  |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|
#                                                  __/ |                    


def generate_histogram():
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    
    ax.set_xlim(0, 10)
    ax.set_xlabel('MAE (cm)')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    
    ax.set_ylim(0, 35)
    ax.set_ylabel('Number of Taxels')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    
    metadata = [
        
        {
            'tool_length'     : 20,
            'loss_alltool'    : 1.2053,
            'loss_halftool'   : 1.4317,
            'color_histogram' : 'mistyrose',
            'color_alltool'   : 'orangered',
            'color_halftool'  : 'lightcoral'
        },
        
        {
            'tool_length'     : 30,
            'loss_alltool'    : 2.3250,
            'loss_halftool'   : 2.6266,
            'color_histogram' : 'honeydew',
            'color_alltool'   : 'darkgreen',
            'color_halftool'  : 'limegreen'
        },
        
        {
            'tool_length'     : 50,
            'loss_alltool'    : 3.4561,
            'loss_halftool'   : 3.7915,
            'color_histogram' : 'lightcyan',
            'color_alltool'   : 'royalblue',
            'color_halftool'  : 'deepskyblue'
        }
        
    ]
    
    for metadatum in metadata:
        
        df_losses = pd.read_csv(f'../results/neutouch_singletool_{metadatum["tool_length"]}.csv',
                                names=['taxel', 'test_loss_mean', 'test_loss_std'],
                                dtype={'taxel': int, 'test_loss_mean': float, 'test_loss_std': float})
        
        ax.hist(df_losses.test_loss_mean.values, np.linspace(0, 10, 50), color=metadatum['color_histogram'])
        ax.axvline(x=metadatum['loss_alltool'], color=metadatum['color_alltool'], linewidth=1, linestyle='--', label=f'alltool_{metadatum["tool_length"]}')
        ax.axvline(x=metadatum['loss_halftool'], color=metadatum['color_halftool'], linewidth=1, linestyle='--', label=f'halftool_{metadatum["tool_length"]}')
    
    ax.legend()
    fig.savefig(f'neutouch_singletool_histogram.pdf', bbox_inches='tight')


#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      


if __name__ == '__main__':
    
    mpl.rcParams.update({'font.size': 6})
    generate_heatmap()
    generate_histogram()
