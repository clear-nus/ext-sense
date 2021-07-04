# Instructions to generate kernel features from raw spikes

There are 3 tasks (task):(tool_type)

1. Tapping on rod (rodTap):(20,30,50)
2. Handover (handover):(0,1,2)
3. Food Poking (foodPoking):(-1)

The data can be sampled at different frequencies:
freq = {4000, 3000, 2000, 1000, 500, 200, 100, 50, 10, 5}

```
python convolute.py --data_dir '{data_dir}' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/convoluted/' --task {task} --frequency {freq} --tool_ty
```

Note: to generate features from kernel based convolution, we use pretrained kernels from 

https://github.com/uzh-rpg/rpg_event_representation_learning