from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataloader import default_collate
import pandas as pd

# DATASET

class RawTactileDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.samples = np.load(Path(path) / f'all.npy')
        self.size=self.samples.shape[0]

    def __getitem__(self, index):
        tact = np.load(Path(self.path)/( str(int(self.samples[index,0])) +'.npy') )
        if tact.shape[0] == 0:
            print('Warning!', str(int(self.samples[index,0])) +'.npy')
        # normalize
        tact[:,1] -= tact[0,1]
        tact[:,1] /= tact[:,1].max()
        return (
            torch.FloatTensor( tact ), # data
            torch.FloatTensor( [self.samples[index,3]] ), # label
        )

    def __len__(self):
        return self.size
    
    
# LOADER

class TacLoader:
    def __init__(self, dataset, batch_size, device, num_workers=4, pin_memory=True, shuffle=False):
        self.device = device
        
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                         num_workers=num_workers, pin_memory=pin_memory,
                                         collate_fn=collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    return events, labels