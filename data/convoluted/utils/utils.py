import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import tqdm

def read_spikes(file_path, time_zero=False, return_start=False):
    raw_df = pd.read_csv(file_path,names=['isPos', 'taxel', 'removable','t'], sep=' ',
                        dtype={'isPos': np.int , 'taxel':np.int, 'removable':np.int,'t':np.float})
    raw_df = raw_df.drop(['removable'], axis=1)

    start_time = raw_df.t[0]
    raw_df.drop(raw_df.tail(1).index,inplace=True)
    raw_df.drop(raw_df.head(1).index,inplace=True)
    
    if time_zero:
        raw_df.t -= raw_df.t.values[0]
    
    raw_df = raw_df.reset_index(drop=True)
    
    if return_start:
        return raw_df, start_time
    
    return raw_df

def downsample_raw_spikes(df, frequency, max_frequency=4000):
    
    assert frequency <= 4000, f'Frequency cannot be higher than {max_frequency}'
    
    if frequency == max_frequency:
        return df

    print(f'Downsampling at {frequency} ...')
    
    last_spiked = {}
    for i in range(1, 81):
        last_spiked[i] = 0.0
    indics= []
    for i, row in df.iterrows():
        if int(row.taxel) <1 or int(row.taxel)> 80:
            continue
        if (row.t - last_spiked[int(row.taxel)]) >= 1/frequency:
            indics.append(i)
            last_spiked[int(row.taxel)] = row.t
            
    downsampled_df = df.iloc[indics]
    downsampled_df = downsampled_df.reset_index(drop=True)
     
    return downsampled_df


def prepare_rod_raw_spikes(data_dir, save_dir, tool_type, frequency=4000, num_splits=4, feature_t=250, delay_t=50, label_map=None):
    info = []
    count = 0
    trials=10
    for trial in range(1, trials+1) :
        fname = f'trial{trial}_{tool_type}'
        df = read_spikes(data_dir / f'{fname}.tact')
        df = downsample_raw_spikes(df, frequency)
        df_estls = pd.read_csv(data_dir / f'{fname}_essentials.csv')
        contact_times = df_estls.t.values
        labels_y = df_estls.label_y.values
        for label_y, contact_t in zip(labels_y, contact_times):
            sample_df = df[(df.t >= (contact_t-delay_t/1000)) & (df.t < (contact_t+feature_t/1000))]
            sample_values = sample_df[['taxel', 't', 'isPos']].values
            np.save(save_dir / str(count), sample_values)
            info.append([count, trial, tool_type, label_y])
            count +=1
            
    all_info = np.array(info, dtype=float)
    np.save(save_dir / f'all', all_info)
    
    save_split_info(save_dir, all_info, num_splits)
    
    
def prepare_handover_raw_spikes(data_dir, save_dir, tool_type, frequency=4000, num_splits=4, feature_t=250, delay_t=50, label_map=None):
    
    df_estls = pd.read_csv(data_dir / 'nt_essentials.csv')
    df_estls = df_estls.fillna(-1)
    df_estls.obj = df_estls.obj.map(label_map)
    df_estls = df_estls[df_estls.obj == tool_type]
    
    info = []
    count = 0
    for i, row in df_estls.iterrows():
        fname = row.fname
        if fname == 'neg_box_35':
            continue
        df = read_spikes(data_dir / f'{fname}.tact')
        df = downsample_raw_spikes(df, frequency)
        contact_t = row.tapped_time
        sample_df = df[(df.t >= (contact_t-delay_t/1000)) & (df.t < (contact_t+feature_t/1000))]
        sample_values = sample_df[['taxel', 't', 'isPos']].values
        
        info.append([count, 0.0, row.obj, row.isPos,
                     row.label_x_thumb, row.label_y_thumb, row.label_z_index,
                     row.label_x_index, row.label_y_index, row.label_z_index])
        sample_values = sample_df[['taxel', 't', 'isPos']].values
        
        np.save(save_dir / str(count), sample_values)
        count +=1
            
    all_info = np.array(info, dtype=float)
    np.save(save_dir / f'all', all_info)
    
    save_split_info(save_dir, all_info, num_splits)
    
    
def prepare_food_raw_spikes(data_dir, save_dir, tool_type, frequency=4000, num_splits=4, feature_t=250, delay_t=50, label_map=None):
    
    df_estls = pd.read_csv(data_dir / 'nt_essentials.csv', names=['old_dir', 't', 'obj'])
    
    df_estls.obj = df_estls.obj.map(label_map)
    
    def refine_dir(x):
        sub_dirs = x.split('/')
        fname = sub_dirs[-1]
        sub_dir = sub_dirs[-2]
        new_sub_dir = sub_dir + '/' + fname
        return new_sub_dir

    df_estls = df_estls.assign(new_sub_dir = df_estls.old_dir.apply(refine_dir))
    
    info = []
    count = 0
    for i, row in df_estls.iterrows():
        local_data_dir = Path(data_dir) / row.new_sub_dir
        df = read_spikes(local_data_dir)
        df = downsample_raw_spikes(df, frequency)
        contact_t = row.t
        sample_df = df[(df.t >= (contact_t-delay_t/1000)) & (df.t < (contact_t+feature_t/1000))]
        sample_values = sample_df[['taxel', 't', 'isPos']].values
        #print(row.obj)

        info.append([count, 0.0, row.obj, row.obj])
        sample_values = sample_df[['taxel', 't', 'isPos']].values
        np.save(save_dir / str(count), sample_values)

        count +=1

    all_info = np.array(info, dtype=float)
    #print(all_info)
    np.save(save_dir / f'all', all_info)

    #save_split_info(save_dir, all_info, num_splits)
    
def save_split_info(save_dir, all_info, num_splits):
    
    skf = StratifiedKFold(n_splits=num_splits, random_state=100, shuffle=True)
    
    train_indices = []
    test_indices = []

    for train_index, test_index in skf.split(np.zeros(all_info.shape[0]), all_info[:,2]):
        train_indices.append(train_index)
        test_indices.append(test_index)

    for split in range(num_splits):
        np.save(save_dir / f'train_split_{split+1}', all_info[train_indices[split]])
        np.save(save_dir / f'test_split_{split+1}', all_info[test_indices[split]])
    