import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm

# understand how to apply quantization layer
class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9, learnt_kernel_dir=None):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        #path = join('utils', "quantization_layer_init", "trilinear_init.pth")
        if learnt_kernel_dir != None:
            state_dict = torch.load(learnt_kernel_dir)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values
    
    
class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 learnt_kernel_dir=None):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0],
                                      learnt_kernel_dir=learnt_kernel_dir)
        self.dim = dim

    def forward(self, events):
        #print('---------------------------------------------------')
        #print('Quantization layer (events: x,y,t,p,batch_number): ', events.shape)
        B = int((1+events[-1,-1]).item())
        #print('B: ', B)
        #print('dim: ', self.dim)
        num_voxels = int(2 * np.prod(self.dim) * B)
        #print('num_voxels: ', num_voxels)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        #print('vox: ', vox.shape)
        C, H = self.dim

        # get values for each channel
        x, t, p, b = events.t()

        #print('x:', x)
        #print('p:', p)
        # p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + 0 \
                          + 0 \
                          + H * C * p \
                          + H * C * 2 * b - 1
        
        #print('idx_before_bins:', idx_before_bins.sort())

        for i_bin in range(C):
            #print('i_bin:', i_bin)
            #print('substraction:', i_bin/(C-1))
            #print('t-i:', t-i_bin/(C-1))
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            #print('values:', values)
            
            # draw in voxel grid
            idx = idx_before_bins + H * i_bin
            #print('idx:', idx)
            vox.put_(idx.long(), values, accumulate=True)
            
            #print('vox section:', vox)

        #print('vox:', vox.shape)
        vox = vox.view(-1, 2, C, H)
        #print('final vox 1:', vox.shape)
        vox = torch.transpose(vox, 1,2)
        vox = vox.reshape(-1,C,160)
        #vox = torch.transpose(vox, 0,1) # return time, batch, 160
        vox /= 100
        #print('final vox 2:', vox.shape)
        return vox