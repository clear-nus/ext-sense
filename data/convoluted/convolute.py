# preprocess kernel features
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
from utils.utils import prepare_rod_raw_spikes, prepare_handover_raw_spikes, prepare_food_raw_spikes
from utils.kernels import QuantizationLayer
from utils.dataset import RawTactileDataset, TacLoader
import torch.nn as nn
import tqdm

parser = argparse.ArgumentParser(description="Kernel data preprocessor.")

parser.add_argument(
    "--save_dir_raw_spikes", type=str, help="", required=True
)

parser.add_argument(
    "--save_dir_kernel", type=str, help="Data save path.", required=True
)

parser.add_argument(
    "--data_dir", type=str, help="Path to dataset.", required=True
)

parser.add_argument(
    "--task",
    type=str,
    help="Experiment name for different tasks.",
    choices=["rodTap", "handover", "foodPoking"],
    required=True,
)

parser.add_argument("--n_jobs", type=int, help="Number of threads used for kernel preprocess.", default=36)
parser.add_argument("--frequency", type=int, help="Downsampling frequency: min=1, max=4000.", default=4000)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=1)

parser.add_argument(
    "--FUTURE_T", type=float, help="Ahead time after tapping happened.", default=250
)
parser.add_argument(
    "--DELAY_T", type=float, help="Back time before tapping happened.", default=50
)

parser.add_argument(
    "--device", type=str, help="Cuda device name", default='cuda'
)

parser.add_argument(
    "--tool_type",
    type=int,
    help="Experimental objects.",
    choices=[20,30,50, 0, 1, 2, -1],
    required=True,
)

parser.add_argument(
    "--num_splits",
    type=int,
    help="Number of splits for stratified K-folds.",
    default=4,
)

args = parser.parse_args()

METHOD='kernel'

# understand which task to do
tasks = {'handover':[0, 1, 2],
         'rodTap':[20,30,50],
         'foodPoking':[-1]}

handover_map = {'rod': 0, 'plate': 1, 'box': 2 }
food_map = {'empty': 0, 'water': 1, 'tofu': 2, 'banana':3, 'watermelon':4, 'apple': 5, 'pepper': 6}

 # check task 
assert args.task in tasks.keys(), f'Task {args.task} is not found'
assert args.tool_type in tasks[args.task] , f'tool_type {args.tool_type} is undefined given in {args.task} task'

# food poking does not have types (classification task): simple workaround to name it as task
if args.task == 'foodPoking':
    args.tool_type = args.task

    
# generalized dir/file name
fname = f'{METHOD}_{args.task}_{args.tool_type}_{args.frequency}'
raw_spikes_dir = Path(args.save_dir_raw_spikes) / fname
raw_spikes_dir.mkdir(parents=True, exist_ok=True)

# locate all tactile files



#print(args.FUTURE_T)
print(f'Preparing to create raw spikes for object type {args.tool_type} in task {args.task}.')
if args.task == 'rodTap':
    prepare_rod_raw_spikes(data_dir=Path(args.data_dir),
                           save_dir=raw_spikes_dir,
                           tool_type=args.tool_type,
                           frequency=args.frequency,
                           num_splits = args.num_splits,
                           feature_t=args.FUTURE_T,
                           delay_t=args.DELAY_T,
                           label_map=None)
elif args.task == 'handover':
    prepare_handover_raw_spikes(data_dir=Path(args.data_dir),
                           save_dir=raw_spikes_dir,
                           tool_type=args.tool_type,
                           frequency=args.frequency,
                           num_splits = args.num_splits,
                           feature_t=args.FUTURE_T,
                           delay_t=args.DELAY_T,
                           label_map=handover_map)
elif args.task == 'foodPoking':
    ### CHECK!
    prepare_food_raw_spikes(data_dir=Path(args.data_dir),
                           save_dir=raw_spikes_dir,
                           tool_type=args.tool_type,
                           frequency=args.frequency,
                           num_splits = args.num_splits,
                           feature_t=args.FUTURE_T,
                           delay_t=args.DELAY_T,
                           label_map=food_map)

print(f'Done with processing with raw spikes.')

print(f'Preparing to create kernel features for object type {args.tool_type} in task {args.task}.')
device = torch.device(args.device)
model = QuantizationLayer((50,80), [1, 30, 30, 1],
                          nn.LeakyReLU(negative_slope=0.1),learnt_kernel_dir='utils/trilinear_init.pth').to(device)
model.eval()

tactile_dataset = RawTactileDataset(str(raw_spikes_dir))
# construct loader, handles data streaming to gpu
tactile_loader = TacLoader(tactile_dataset, batch_size=args.batch_size, device=device)

all_representation = []
targets = []
count = 0
for events, label in tqdm.tqdm(tactile_loader):
    representation = model(events).detach().cpu().numpy()
    all_representation.append(representation)
    targets.append(label.cpu().numpy())
    count += 1
#     if count == 10:
#         break
    
all_representation = np.vstack(all_representation)
targets = np.vstack(targets)
#print(all_representation.shape)
#print(targets.shape)


kernel_dir = Path(args.save_dir_kernel)
np.savez(kernel_dir / f'{fname}.npz', signals=all_representation, labels=targets)

print(f'Done with generating features using kernel.')
