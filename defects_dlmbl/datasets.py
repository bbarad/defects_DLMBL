# conda install jupyter pytorch tensorboard h5py
import h5py
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import zarr

def create_data(
    url, 
    name, 
    offset, 
    resolution,
    sections=None,
    squeeze=True):

  in_f = h5py.File(io.BytesIO(requests.get(url).content), 'r')

  raw = in_f['volumes/raw']
  labels = in_f['volumes/labels/neuron_ids']
  
  f = zarr.open(name, 'a')

  if sections is None:
    sections=range(raw.shape[0]-1)

  for i, r in enumerate(sections):

    print(f'Writing data for section {r}')

    raw_slice = raw[r:r+1,:,:]
    labels_slice = labels[r:r+1,:,:]

    if squeeze:
      raw_slice = np.squeeze(raw_slice)
      labels_slice = np.squeeze(labels_slice)

    f[f'raw/{i}'] = raw_slice
    f[f'labels/{i}'] = labels_slice

    f[f'raw/{i}'].attrs['offset'] = offset
    f[f'raw/{i}'].attrs['resolution'] = resolution

    f[f'labels/{i}'].attrs['offset'] = offset
    f[f'labels/{i}'].attrs['resolution'] = resolution

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


class CREMIDataset(Dataset):
    def __init__(self,filename):
        self.filename = filename
        self.samples = self.get_num_samples()

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        with zarr.open(self.filename, 'r') as f:
            samples = len(list(f['raw']))
        return samples

    def augment_image_and_labels(self,x,y):
        x = np.expand_dims(x, 0)
        x_len = 1
        stack = np.concatenate((x,y), axis=0)
        augmenter = iaa.Sequential([iaa.CropToFixedSize(height=256, width=256)])
        stack = np.array(augmenter.augment_images(stack))
        x,y = stack[:x_len,...], stack[x_len:,...]
        return x,y

    def affinities(self, y, pad_size=1):
        # shift 1
        y0 = y[:-pad_size,:]
        y1 = y[pad_size:, :]
        aff1 = y0==y1
        aff1 = np.pad(aff1, ((1,0), (0,0)))
        
        #shift 2
        y11 = y[:, :-pad_size]
        y12 = y[:, pad_size:]
        aff2 = y11==y12
        aff2 = np.pad(aff2, ((0,0), (1,0)))
        
        return np.stack((aff1, aff2), axis=0)
            
    def __getitem__(self,index):
        with zarr.open(self.filename, 'r') as test:
            x = test[f'raw/{index}'][...]
            y = test[f'labels/{index}'][...]
            y = self.affinities(y)
            x,y = self.augment_image_and_labels(x,y)

            x = torch.tensor(x).float()
            y = torch.tensor(y)

        return x, y