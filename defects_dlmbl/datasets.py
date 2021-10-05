import io
import os
import requests

from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
import glob
from glob import glob
import h5py
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import skimage
import tifffile
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
        self.filenames = glob(filename) # works with glob strings, but have to pass list of indices if you want to specify indices
        self.offsets=offsets   
        if indices is None:
            indices = [list(range(self.get_num_samples(f))) for f in self.filenames]
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

    def get_num_samples(self,filename):
        with zarr.open(filename, 'r') as f:
            samples = len(list(f['raw']))
        return samples

    def read_data(self,indices):
        xs = []
        ys = []
        for f,idxs in zip(self.filenames,indices):
            with zarr.open(f, 'r') as z:
                xs.extend([z[f'raw/{index}'] for index in idxs])
                ys.extend([z[f'labels/{index}'] for index in idxs])
        x = np.array(xs)
        y = np.array(ys)
        return x,y
    

    def augment_image_and_labels(self,x,y):
        assert len(x.shape)==2 and len(y.shape)==2
        x = x[...,None]
        y = y[None,...,None]
        ia.seed(np.random.randint(1, 100000))
        x,y = self.augmenter(image=x, segmentation_maps=y)
        cropper = iaa.Sequential([iaa.CropToFixedSize(height=cropsize, width=cropsize)])
        x,y = cropper(image=x, segmentation_maps=y)
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

class NMJDataset2(Dataset):
    def __init__(self,data_path,indices=None,offsets=[[-1, 0], [0, -1], [-9, 0], [0, -9]], augmenter=None, augment_and_crop=True, pad=0, crop_size=None):
        
        self.data_path = data_path
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

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        samples = len(glob.glob(os.path.join(self.data_path,"Raw_Norm/*.tif")))
        return samples

    def read_data(self,indices):
        x_list = glob.glob(os.path.join(self.data_path+"Raw_Norm/*.tif"))
        y_list = glob.glob(os.path.join(self.data_path+"GT_Norm/*.tif"))
        x_list.sort()
        y_list.sort()
        
        x = []
        y = []
        for ix in indices:
            x.append(tifffile.imread(x_list[ix]).transpose())
            y.append(tifffile.imread(y_list[ix]).transpose())
            
        return x, y
    
    def augment_image_and_labels(self,x,y, cropsize=None):
        cropsize = self.crop_size
        ia.seed(np.random.randint(1, 100000))
        x,y = self.augmenter(image=x, segmentation_maps=y)
        if cropsize:
            cropper = iaa.CropToFixedSize(height=cropsize, width=cropsize)
            x,y = cropper(image=x, segmentation_maps=y)
        x = np.moveaxis(x, 2,0)
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
        # x = np.expand_dims(x, 0)
        
        y = skimage.measure.label(y).astype('int16')
        y = np.expand_dims(y, -1)
        y = np.expand_dims(y, 0)
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
            y = np.pad(y,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        
        aff = self.affinities(y)
        x = torch.tensor(x.astype(np.int16)).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y


class WingDataset(Dataset):
    def __init__(self,data_path,indices=None,offsets=[[-1, 0], [0, -1], [-5, 0], [0, -5]], augmenter=None, augment_and_crop=True, pad=0, crop_size=None):
        
        self.data_path = data_path
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

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        trainval_data =  np.load('/home/delsignores/defects_DLMBL/data/Flywing_n0/train/train_data.npz')
        train_images = trainval_data['X_train'].astype(np.float32)
        train_masks = trainval_data['Y_train']
        val_images = trainval_data['X_val'].astype(np.float32)
        val_masks = trainval_data['Y_val']
        x = np.concatenate((train_images, val_images))
        samples = len(x)
        return samples

    def read_data(self,indices):
        trainval_data =  np.load('/home/delsignores/defects_DLMBL/data/Flywing_n0/train/train_data.npz')
        train_images = trainval_data['X_train'].astype(np.float32)
        train_masks = trainval_data['Y_train']
        val_images = trainval_data['X_val'].astype(np.float32)
        val_masks = trainval_data['Y_val']
        
        x = np.concatenate((train_images, val_images))
        y = np.concatenate((train_masks, val_masks))
        return x, y
    
    def augment_image_and_labels(self,x,y, cropsize=None):
        cropsize = self.crop_size
        ia.seed(np.random.randint(1, 100000))
        x,y = self.augmenter(image=x, segmentation_maps=y)
        if cropsize:
            cropper = iaa.CropToFixedSize(height=cropsize, width=cropsize)
            x,y = cropper(image=x, segmentation_maps=y)
        y = y[0,...,0]
        x = np.moveaxis(x,2,0)
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
        x = np.expand_dims(x, -1)
        
        y = skimage.measure.label(y).astype('int16')
        y = np.expand_dims(y, -1)
        y = np.expand_dims(y, 0)
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
            y = np.pad(y,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        
        aff = self.affinities(y)
        x = torch.tensor(x.astype(np.int16)).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y


class WingTestDataset(Dataset):
    def __init__(self,data_path,indices=None,offsets=[[-1, 0], [0, -1], [-5, 0], [0, -5]], augmenter=None, augment_and_crop=True, pad=0, crop_size=None):
        
        self.data_path = data_path
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

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        trainval_data =  np.load('/home/delsignores/defects_DLMBL/data/Flywing_n0/test/test_data.npz')
        train_images = trainval_data['X_test'].astype(np.float32)
        train_masks = trainval_data['Y_test']
        samples = len(train_images)
        return samples

    def read_data(self,indices):
        trainval_data =  np.load('/home/delsignores/defects_DLMBL/data/Flywing_n0/test/test_data.npz')
        train_images = trainval_data['X_test'].astype(np.float32)
        train_masks = trainval_data['Y_test']
        return x, y
    
    def augment_image_and_labels(self,x,y, cropsize=None):
        cropsize = self.crop_size
        ia.seed(np.random.randint(1, 100000))
        x,y = self.augmenter(image=x, segmentation_maps=y)
        if cropsize:
            cropper = iaa.CropToFixedSize(height=cropsize, width=cropsize)
            x,y = cropper(image=x, segmentation_maps=y)
        y = y[0,...,0]
        x = np.moveaxis(x,2,0)
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
        x = np.expand_dims(x, -1)
        
        y = skimage.measure.label(y).astype('int16')
        y = np.expand_dims(y, -1)
        y = np.expand_dims(y, 0)
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
            y = np.pad(y,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        
        aff = self.affinities(y)
        x = torch.tensor(x.astype(np.int16)).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y


class NMJDataset(Dataset):
    def __init__(self,data_path,indices=None,offsets=[[-1, 0], [0, -1], [-9, 0], [0, -9]], augmenter=None, augment_and_crop=True, pad=0, crop_size=None):
        
        self.data_path = data_path
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

    def __len__(self):
        return self.samples

    def get_num_samples(self):
        samples = len(glob.glob(os.path.join(self.data_path,"Raw_Norm/*.tif")))
        return samples

    def read_data(self,indices):
        x_list = glob.glob(os.path.join(self.data_path+"Raw_Norm/*.tif"))
        y_list = glob.glob(os.path.join(self.data_path+"GT_Norm/*.tif"))
        x_list.sort()
        y_list.sort()
        
        x = []
        y = []
        for ix in indices:
            x.append(tifffile.imread(x_list[ix]).transpose())
            y.append(tifffile.imread(y_list[ix]).transpose())
            
        return x, y
    
    def augment_image_and_labels(self,x,y, cropsize=None):
        cropsize = self.crop_size
        ia.seed(np.random.randint(1, 100000))
        x,y = self.augmenter(image=x, segmentation_maps=y)
        if cropsize:
            cropper = iaa.CropToFixedSize(height=cropsize, width=cropsize)
            x,y = cropper(image=x, segmentation_maps=y)
        y = y[0,...,0]
        x = np.moveaxis(x,2,0)
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
        x = np.expand_dims(x, -1)
        
        y = skimage.measure.label(y).astype('int16')
        y = np.expand_dims(y, -1)
        y = np.expand_dims(y, 0)
        if self.pad:
            x = np.pad(x,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
            y = np.pad(y,((self.pad,self.pad),(self.pad,self.pad)),'reflect')
        if self.augment_and_crop:
            x,y = self.augment_image_and_labels(x,y)
        
        aff = self.affinities(y)
        x = torch.tensor(x.astype(np.int16)).float()
        aff = torch.tensor(aff)
        y = torch.tensor(y).unsqueeze(0)
        return x, aff, y
