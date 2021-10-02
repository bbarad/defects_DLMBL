# conda install jupyter pytorch tensorboard h5py

import torch
from torch.utils.data import Dataset
import h5py


class ISBIDataset(Dataset):
    def __init__(self,filename):
        self.filename = filename
        self.samples = self.get_num_samples()

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        with h5py.File(self.filename) as f:
            samples = f['raw'].shape[0]
        return samples

    def __getitem__(self,index):
        with h5py.File(self.filename) as f:
            x = f['raw'][index]
            y = f['affinities'][0,index]
        return torch.tensor(x).unsqueeze(0),torch.tensor(y).long()

