import io
import requests

from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
import h5py
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset
import zarr
from scipy import ndimage
import torch.nn.functional as F

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
    def __init__(self,filename,indices=None,offsets=[[-1, 0], [0, -1], [-9, 0], [0, -9]], augmenter=None, augment_and_crop=True, pad=0, crop_size=260):

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
        self.crop_size = crop_size
        self.cropper = iaa.Sequential([iaa.CropToFixedSize(height=self.crop_size, width=self.crop_size)])

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

    def augment_image_and_labels(self,x,y):
        assert len(x.shape)==2 and len(y.shape)==2
        x = x[...,None]
        y = y[None,...,None]
        ia.seed(np.random.randint(1, 100000))
        
        x,y = self.cropper(image=x, segmentation_maps=y)
        x,y = self.augmenter(image=x, segmentation_maps=y)
       
        x = x[None,...,0]
        y = y[0,...,0]
        return x,y

    def affinities(self,y):
        seg2aff = Segmentation2AffinitiesWithPadding(
            self.offsets,
            retain_segmentation=False,
            segmentation_to_binary=False,
            ignore_label=-1)
        return 1-seg2aff.tensor_function(y)
            
    def __getitem__(self,index):
        x = self.x[index]
        y = self.y[index]
        y = skimage.measure.label(y).astype('int16')
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        aff = self.affinities(y)
        x = torch.tensor(x).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y


class TomoDataset(CREMIDataset):
    """Dataset handler for a 2D affinity training network for tomo data"""
    def __init__(self,filename,validation=False,validation_set=[],offsets=[[-1, 0], [0, -1], [-2, 0], [0, -2]], augmenter=None, augment_and_crop=True, pad=0, crop_size=516): # 924
        self.filename = filename
        self.data = zarr.open(self.filename, 'r')
        self.index_map = []
        print("Inspecting label fields for views containing ground truth")
        for name, entry in self.data['labels'].arrays():
            for i in range(entry.shape[0]):
                if np.sum(entry[i]) > 50:
                    self.index_map.append((name, i))
            # self.index_map.extend([(name, i) for i in range(entry.shape[0])])
        if validation:
            self.index_map = list(filter(lambda x: x[0] in validation_set, self.index_map))
        else:
            self.index_map = list(filter(lambda x: x[0] not in validation_set, self.index_map))

        self.offsets=offsets   
        self.augment_and_crop = augment_and_crop
        self.pad = pad
        if augmenter:
            self.augmenter = augmenter
        else:
            self.augmenter = iaa.Identity()
        self.crop_size = crop_size
        self.cropper = iaa.Sequential([iaa.CropToFixedSize(height=self.crop_size, width=self.crop_size)])
    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self,index):
        map_index = self.index_map[index]
        # print(map_index)
        x = self.data["data"][map_index[0]][map_index[1]]
        y = self.data["labels"][map_index[0]][map_index[1]]
        # print(x.shape,y.shape)

        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')

        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)

        aff = self.affinities(y)

        x = torch.tensor(x).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        # print(x.shape, aff.shape, y.shape)
        return x, aff, y

class TomoDatasetSemanticDistance(TomoDataset):
    """Dataset handler for a 2D semantic segmentationnetwork for tomo data"""
    def __init__(self,filename,validation=False,validation_set=[], augmenter=None, augment_and_crop=True, pad=0, crop_size=516): # 924
        super().__init__(filename,validation,validation_set,augmenter,augment_and_crop,pad,crop_size)
        self.crop_size = crop_size
        self.cropper = iaa.Sequential([iaa.CropToFixedSize(height=self.crop_size, width=self.crop_size)])
        self.augment_and_crop = augment_and_crop
        self.augmenter=augmenter
        if self.augmenter==None:
            self.augmenter = iaa.Identity()
            print("No augmenter specified, using identity")
    def augment_image_and_labels(self,x,y):
        assert len(x.shape)==2 and len(y.shape)==2
        x = x[...,None]
        y = y[None,...,None]
        ia.seed(np.random.randint(1, 100000))
        
        x,y = self.cropper(image=x, segmentation_maps=y)
        x,y = self.augmenter(image=x, segmentation_maps=y)
       
        x = x[None,...,0]
        y = y[0,...,0]
        return x,y

    def __getitem__(self,index):
        map_index = self.index_map[index]
        # print(map_index)
        x = self.data["data"][map_index[0]][map_index[1]]
        y = self.data["labels"][map_index[0]][map_index[1]]
        # Binarize and cast into int
        y = y>0.5
        # y = y.astype('int16')
        # if self.pad:
        #     x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        y = y.astype('float')
        if y.max()==0:
            dist = y
        # signed distance transform:
        else:
            positive_distance = ndimage.distance_transform_edt(y)
            negative_distance = -1*(ndimage.distance_transform_edt(1-y))
            dist = np.clip(positive_distance+negative_distance, -10, 10)
        y = np.stack((y,dist))
        return x, y