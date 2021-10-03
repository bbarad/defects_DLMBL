import io
import requests

from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
import h5py
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset
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
            y = f['affinities'][:,index]
        return torch.tensor(x).unsqueeze(0),torch.tensor(y)#.long()


class CREMIDataset(Dataset):
    def __init__(self,filename,indices=None,offsets=[[-1, 0], [0, -1], [-9, 0], [0, -9]], augmenter=None, augment_and_crop=True, pad=0):

        self.filename = filename
        # maybe make offsets more flexible. 
        # Could use len(offsets) to set network output channels?
        self.offsets=offsets   
        if indices is None:
            indices = list(range(self.get_num_samples()))
        self.x, self.y = self.read_data(indices)
        self.samples = len(self.x)
        self.augment_and_crop = augment_and_crop
        self.pad = pad
        if augmenter:
            self.augmenter = augmenter
        else:
            self.augmenter = iaa.Identity()

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        with zarr.open(self.filename, 'r') as f:
            samples = len(list(f['raw']))
        return samples

    def read_data(self,indices):
        with zarr.open(self.filename, 'r') as z:
            x = np.array([z[f'raw/{index}'] for index in indices])
            y = np.array([z[f'labels/{index}'] for index in indices])
        return x,y

    def augment_image_and_labels(self,x,y, cropsize=260):
        assert len(x.shape)==2 and len(y.shape)==2
        x = x[...,None]
        y = y[None,...,None]
        
        x,y = self.augmenter(image=x, segmentation_maps=y)
        cropper = iaa.Sequential([iaa.CropToFixedSize(height=cropsize, width=cropsize)])
        x,y = cropper(image=x, segmentation_maps=y)
        x = x[None,...,0]
        y = y[0,...,0]
        return x,y

    def affinities(self,y):
        seg2aff = Segmentation2AffinitiesWithPadding(
            self.offsets,
            retain_segmentation=False,
            segmentation_to_binary=False)
        return seg2aff.tensor_function(y)
            
    def __getitem__(self,index):
        x = self.x[index]
        y = self.y[index]
        y = skimage.measure.label(y).astype('int16')
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
            # y = np.pad(y,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        aff = self.affinities(y)
        x = torch.tensor(x).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y
