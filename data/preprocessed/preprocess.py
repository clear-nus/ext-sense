from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
import tqdm

def preprocess(dataset, data_directory, frequency):
    
    preprocessor_nuskin = {
        'nuskin_tool_20'         : partial(_preprocess_nuskin_tool, 20),
        'nuskin_tool_30'         : partial(_preprocess_nuskin_tool, 30),
        'nuskin_tool_50'         : partial(_preprocess_nuskin_tool, 50),
        'nuskin_handover_rod'    : partial(_preprocess_nuskin_handover, 'rod'),
        'nuskin_handover_box'    : partial(_preprocess_nuskin_handover, 'box'),
        'nuskin_handover_plate'  : partial(_preprocess_nuskin_handover, 'plate'),
        'nuskin_food_apple'      : partial(_preprocess_nuskin_food, 'apple'),
        'nuskin_food_banana'     : partial(_preprocess_nuskin_food, 'banana'),
        'nuskin_food_empty'      : partial(_preprocess_nuskin_food, 'empty'),
        'nuskin_food_pepper'     : partial(_preprocess_nuskin_food, 'pepper'),
        'nuskin_food_tofu'       : partial(_preprocess_nuskin_food, 'tofu'),
        'nuskin_food_water'      : partial(_preprocess_nuskin_food, 'water'),
        'nuskin_food_watermelon' : partial(_preprocess_nuskin_food, 'watermelon')
    }
        
    preprocessor_biotac = {
        'biotac_tool_20'           : partial(_preprocess_biotac_tool, 20),
        'biotac_tool_30'           : partial(_preprocess_biotac_tool, 30),
        'biotac_tool_50'           : partial(_preprocess_biotac_tool, 50),
        'biotac_handover_rod'      : partial(_preprocess_biotac_handover, 'rod'),
        'biotac_handover_box'      : partial(_preprocess_biotac_handover, 'box'),
        'biotac_handover_plate'    : partial(_preprocess_biotac_handover, 'plate'),
        'biotac_food_apple'        : partial(_preprocess_biotac_food, 'apple'),
        'biotac_food_banana'       : partial(_preprocess_biotac_food, 'banana'),
        'biotac_food_empty'        : partial(_preprocess_biotac_food, 'empty'),
        'biotac_food_pepper'       : partial(_preprocess_biotac_food, 'pepper'),
        'biotac_food_tofu'         : partial(_preprocess_biotac_food, 'tofu'),
        'biotac_food_water'        : partial(_preprocess_biotac_food, 'water'),
        'biotac_food_watermelon'   : partial(_preprocess_biotac_food, 'watermelon')
    }

    if dataset in preprocessor_nuskin:
        signals, labels = preprocessor_nuskin[dataset](data_directory, frequency=frequency)
        
    elif dataset in preprocessor_biotac:
        signals, labels = preprocessor_biotac[dataset](data_directory, frequency=frequency)
        
    else: raise Exception('Dataset not found')
    
    np.savez(f'{dataset}_{frequency}hz.npz', signals=signals, labels=labels)
    print(f'Preprocessing for {dataset} for {frequency} Hz completed with signals.shape = {signals.shape} and labels.shape = {labels.shape}')


#    _   _            _                   _        _____                                                       
#   | \ | |          | |                 | |      |  __ \                                                      
#   |  \| | ___ _   _| |_ ___  _   _  ___| |__    | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___  ___  _ __ 
#   | . ` |/ _ \ | | | __/ _ \| | | |/ __| '_ \   |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
#   | |\  |  __/ |_| | || (_) | |_| | (__| | | |  | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ (_) | |   
#   |_| \_|\___|\__,_|\__\___/ \__,_|\___|_| |_|  |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/\___/|_|   
#                                                                | |                                           


def _preprocess_nuskin_tool(tool_length, data_directory, time_past=0.05, time_future=0.25, time_interval=0.005, frequency=4000):
    
    signals = None
    labels = None
        
    print(f'Preprocessing nuskin tool_{tool_length:02d} for {frequency} Hz ...')

    for trial in tqdm.tqdm(range(1, 11)):
        
        df_essentials = pd.read_csv(Path(data_directory)/f'nuskin/tool_1k/trial{trial}_{tool_length}_essentials.csv')
        df_raw = _read_nuskin_raw(Path(data_directory)/f'nuskin/tool_1k/trial{trial}_{tool_length}.tact')
        
        signals_temp = _bin_nuskin_signal(df_essentials.t.values, df_raw, time_past, time_future, time_interval, frequency)
        labels_temp = df_essentials.label_y.values
        
        signals = np.append(signals, signals_temp, axis=0) if signals is not None else signals_temp
        labels = np.append(labels, labels_temp, axis=0) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_nuskin_handover(item, data_directory, time_past=0.05, time_future=0.25, time_interval=0.005, frequency=4000):
    
    signals = None
    labels = None
    
    df_essentials = pd.read_csv(Path(data_directory)/f'nuskin/handover/nt_essentials.csv')
    df_essentials = df_essentials[df_essentials.obj == item]

    print(f'Preprocessing  nuskin handover {item} for {frequency} Hz ... ')

    for _, row in tqdm.tqdm( df_essentials.iterrows(), total=df_essentials.shape[0]  ):
                
        df_raw = _read_nuskin_raw(Path(data_directory)/f'nuskin/handover/{row.fname}.tact')
        tap_time = row.tapped_time
        
        signals_temp = _bin_nuskin_signal(np.array([tap_time]), df_raw, time_past, time_future, time_interval, frequency)
        labels_temp = row[['isPos', 'label_x_thumb', 'label_y_thumb', 'label_z_thumb', 'label_x_thumb_d', 'label_y_thumb_d', 'label_z_thumb_d', 'label_x_index', 'label_y_index', 'label_z_index', 'label_x_index_d', 'label_y_index_d', 'label_z_index_d']].values.astype('float')
        
        signals = np.vstack((signals, signals_temp)) if signals is not None else signals_temp
        labels = np.vstack((labels, labels_temp)) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_nuskin_food(item, data_directory, time_past=0.0, time_future=6.0, time_interval=0.05, frequency=4000):
    
    import glob
    
    signals = None
    labels = None

    print(f'Preprocessing  nuskin food {item} for {frequency} Hz ... ')
    
    for filename in tqdm.tqdm( Path(Path(data_directory)/f'nuskin/food/food_poking_batch1/').glob(f'{item}_zero_*[0-9].tact') ):
        
        df_raw = _read_nuskin_raw(filename)
        start_time = df_raw['t'][0] + 2
        signal_temp = _bin_nuskin_signal(np.array([start_time]), df_raw, time_past, time_future, time_interval, frequency)
        signals = np.vstack((signals, signal_temp)) if signals is not None else signal_temp
    
    labels = np.ones(signals.shape[0])
    
    return signals, labels


def _read_nuskin_raw(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'removable', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(['removable'], axis=1, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    return df.reset_index(drop=True)


def _bin_nuskin_signal(tap_times, df_raw, time_past, time_future, time_interval, frequency):

    n_bins = int((time_past + time_future) / time_interval) + 1
    signals = np.zeros([len(tap_times), 80, n_bins], dtype=int)

    summer = 0
    vals = np.array([])
    
    for i, tap_time in enumerate(tap_times):
        
        df_timespan = df_raw[(df_raw.t >= (tap_time - time_past)) & (df_raw.t < (tap_time + time_future))]
        df_timespan = df_timespan.reset_index(drop=True)
        
        last_spiked = np.zeros(80)
        indices = []
        
        for j, sample in df_timespan.iterrows():
            
            if (sample.t - last_spiked[int(sample.taxel) - 1]) >= 1 / frequency:
                
                indices.append(j)
                last_spiked[int(sample.taxel) - 1] = sample.t
        
        df_timespan = df_timespan.iloc[indices]
        df_timespan = df_timespan.reset_index(drop=True)
        
        df_positive = df_timespan[df_timespan.isPos == 1]
        df_negative = df_timespan[df_timespan.isPos == 0]

        t = tap_time - time_past
        k = 0

        while t < (tap_time + time_future):
            
            positive_taxels = df_positive[((df_positive.t >= t) & (df_positive.t < t + time_interval))].taxel
            if len(positive_taxels):
                for taxel in positive_taxels:
                    signals[i, taxel - 1, k] += 1
                    
            negative_taxels = df_negative[((df_negative.t >= t) & (df_negative.t < t + time_interval))].taxel
            if len(negative_taxels):
                for taxel in negative_taxels:
                    signals[i, taxel - 1, k] -= 1
                    
            t += time_interval
            k += 1
    
    return signals


#    ____  _       _                _____                                                       
#   |  _ \(_)     | |              |  __ \                                                      
#   | |_) |_  ___ | |_ __ _  ___   | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___  ___  _ __ 
#   |  _ <| |/ _ \| __/ _` |/ __|  |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
#   | |_) | | (_) | || (_| | (__   | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ (_) | |   
#   |____/|_|\___/ \__\__,_|\___|  |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/\___/|_|   
#                                                 | |                                           


def _preprocess_biotac_tool(tool_length, data_directory, samples_past=100, samples_future=500, frequency=2200):
    
    signals = None
    labels = None
        
    print(f'Preprocessing biotac_tool_{tool_length} for {frequency} Hz ... ')

    for trial in tqdm.tqdm( range(1, 21) ):
        
        df_essentials = pd.read_csv(Path(data_directory)/f'biotac/tool_1k/trial{trial}_{tool_length}_essentials.csv')
        df_raw = pd.read_csv(Path(data_directory)/f'biotac/tool_1k/trial{trial}_{tool_length}.csv')
        
        signals_trial = _crop_biotac_signal(df_essentials.orignal_index.values, df_raw, samples_past, samples_future)
        labels_trial = df_essentials.label_y.values
        
        signals = np.append(signals, signals_trial, axis=0) if signals is not None else signals_trial
        labels = np.append(labels, labels_trial, axis=0) if labels is not None else labels_trial
    
    return _downsample_biotac_signal(signals, frequency), labels


def _preprocess_biotac_handover(item, data_directory, samples_past=100, samples_future=500, frequency=2200):
    
    signals = None
    labels = None
    
    df_essentials = pd.read_csv(Path(data_directory)/f'biotac/handover/bt_essentials.csv')
    df_essentials = df_essentials[df_essentials.obj == item]
    
    print(f'Preprocessing  biotac handover {item} for {frequency} Hz ... ')
    
    for _, row in tqdm.tqdm( df_essentials.iterrows(), total=df_essentials.shape[0] ) :
                
        df_raw = pd.read_csv(Path(data_directory)/f'biotac/handover/{row.fname}.csv')
        tap_index = np.abs(df_raw.t - row.tapped_time).argmin()
        
        signals_temp = _crop_biotac_signal(np.array([tap_index]), df_raw, samples_past, samples_future)
        labels_temp = row[['isPos', 'label_x_thumb', 'label_y_thumb', 'label_z_thumb', 'label_x_thumb_d', 'label_y_thumb_d', 'label_z_thumb_d', 'label_x_index', 'label_y_index', 'label_z_index', 'label_x_index_d', 'label_y_index_d', 'label_z_index_d']].values.astype('float')
        
        signals = np.vstack((signals, signals_temp)) if signals is not None else signals_temp
        labels = np.vstack((labels, labels_temp)) if labels is not None else labels_temp
    
    return _downsample_biotac_signal(signals, frequency), labels


def _preprocess_biotac_food(item, data_directory, samples_past=600, samples_future=600, frequency=2200):
    
    import glob
    
    signals = None
    labels = None
    
    for filename in tqdm.tqdm( Path( Path(data_directory)/f'biotac/food/').glob(f'{item}_zero_*.csv') ):
        
        df_raw = pd.read_csv(filename)
        signal_temp = df_raw.pac.values
        trigger_index = np.argmax(np.abs(signal_temp))
        signal_temp = signal_temp[trigger_index-samples_past:trigger_index+samples_future]

        signals = np.vstack((signals, signal_temp)) if signals is not None else signal_temp
    
    labels = np.ones(signals.shape[0])
    
    return _downsample_biotac_signal(signals, frequency), labels


def _crop_biotac_signal(tap_indices, df_raw, samples_past=100, samples_future=500):
    
    signals = np.zeros((len(tap_indices), samples_past + samples_future))
    
    for i, tap_index in enumerate(tap_indices):
        signals[i] = (df_raw.iloc[tap_index-samples_past:tap_index+samples_future].pac.values)
    
    return signals


def _downsample_biotac_signal(signals, frequency=2200):
    
    if frequency == 2200: return signals
    
    samples_to_keep = int(frequency / 2200 * signals.shape[1]) + 1
    indices = np.round(np.linspace(0, signals.shape[1] - 1, samples_to_keep)).astype(int)
    
    return signals[:, indices]


#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) == 4:
        preprocess(sys.argv[1], sys.argv[2], int(sys.argv[3]))
