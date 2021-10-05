from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from glob import glob

from .datasets import ISBIDataset, CREMIDataset, TomoDataset, TomoDatasetSemanticDistance, NMJDataset, WingDataset


class ISBIDataModule(LightningDataModule):
	def __init__(self, train_filename):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename

	def setup(self, stage = None):
		self.train = ISBIDataset(self.train_filename)

	def train_dataloader(self):
		return DataLoader(self.train,batch_size=1)
	
	# def train_dataloader(self):
	# 	return DataLoader(self.test,batch_size=1)

	# def val_dataloader(self):
	# 	return DataLoader(self.val,batch_size=1)


class CREMIDataModule(LightningDataModule):
	def __init__(self, train_filename, augmenter = None, augment_and_crop=True, pad=0, offsets=[[-1, 0], [0, -1], [-9, 0], [0, -9]]):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename
		self.augmenter = augmenter
		self.augment_and_crop = augment_and_crop
		self.pad = pad
		self.offsets = offsets


	def setup(self, stage = None):
		globbed = glob(self.train_filename) # works with glob strings
		train_indices = []
		val_indices = []
		total = 0
		for f in globbed:
			full_dataset = len(CREMIDataset(f))
			split_index = 4*full_dataset//5
			train_indices.append(list(range(split_index)))
			val_indices.append(list(range(split_index,full_dataset)))
			total+=full_dataset
		
		self.train = CREMIDataset(self.train_filename,indices=train_indices, augmenter=self.augmenter, offsets = self.offsets,
								augment_and_crop=self.augment_and_crop, pad=self.pad)
		self.val = CREMIDataset(self.train_filename,indices=val_indices, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	

	def train_dataloader(self):
		return DataLoader(self.train,batch_size=4, num_workers=8, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val,batch_size=4, num_workers=8)

class NMJDataModule(LightningDataModule):
	def __init__(self, train_filename, augmenter = None, augment_and_crop=True, pad=0, offsets=[[-1, 0], [0, -1], [-5, 0], [0, -5]]):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename
		self.augmenter = augmenter
		self.augment_and_crop = augment_and_crop
		self.pad = pad
		self.offsets = offsets

	def setup(self, stage = None):
		full_dataset = len(NMJDataset(self.train_filename))
		split_index = 4*full_dataset//5
		train_indices = list(range(split_index))
		val_indices = list(range(split_index,full_dataset))
		self.train = NMJDataset(self.train_filename,indices=train_indices, augmenter=self.augmenter, offsets = self.offsets,
								augment_and_crop=self.augment_and_crop, pad=self.pad)
		
		self.val = NMJDataset(self.train_filename,indices=val_indices, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	

		self.test = NMJDataset(self.train_filename, indices=val_indices, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	


	def train_dataloader(self):
		return DataLoader(self.train, batch_size=24, num_workers=8, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val, batch_size=3, num_workers=8)
	
	def test_dataloader(self):
		return DataLoader(self.test, batch_size=8, num_workers=8)


class WingDataModule(LightningDataModule):
	def __init__(self, train_filename, test_filename, augmenter = None, augment_and_crop=True, pad=0, offsets=[[-1, 0], [0, -1], [-5, 0], [0, -5]]):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename
		self.test_filename = test_filename
		self.augmenter = augmenter
		self.augment_and_crop = augment_and_crop
		self.pad = pad
		self.offsets = offsets

	def setup(self, stage = None):
		full_dataset = len(WingDataset(self.train_filename))
		split_index = 4*full_dataset//5
		train_indices = list(range(split_index))
		val_indices = list(range(split_index,full_dataset))
		self.train = WingDataset(self.train_filename,indices=train_indices, augmenter=self.augmenter, offsets = self.offsets,
								augment_and_crop=self.augment_and_crop, pad=self.pad)
		self.val = WingDataset(self.train_filename,indices=val_indices, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	

		self.test = WingDataset(self.test_filename, indices=val_indices, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	

	def train_dataloader(self):
		return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val, batch_size=8, num_workers=8)

	def test_dataloader(self):
		return DataLoader(self.test, batch_size=8, num_workers=8)

class TomoDataModule(CREMIDataModule):
	def setup(self, stage = None):
		full_dataset = len(TomoDataset(self.train_filename))

		val_names = ["TT1", "TE1", "TF1"] 
		self.train = TomoDataset(self.train_filename,validation=False,validation_set=val_names, augmenter=self.augmenter, offsets = self.offsets,
								augment_and_crop=self.augment_and_crop, pad=self.pad)
		self.val = TomoDataset(self.train_filename,validation=True,validation_set=val_names, offsets = self.offsets, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	
	
	def train_dataloader(self):
		return DataLoader(self.train,batch_size=4, num_workers=4, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val,batch_size=4, num_workers=4,)

class TomoDataSemModule(TomoDataModule):
	def setup(self, stage = None):
		full_dataset = len(TomoDataset(self.train_filename))

		val_names = ["TT1", "TE1", "TF1"] 
		self.train = TomoDatasetSemanticDistance(self.train_filename,validation=False,validation_set=val_names, augmenter=self.augmenter,
								augment_and_crop=self.augment_and_crop, pad=self.pad)
		self.val = TomoDatasetSemanticDistance(self.train_filename,validation=True,validation_set=val_names, 
								augment_and_crop=self.augment_and_crop, pad=self.pad)	
	
	def train_dataloader(self):
		return DataLoader(self.train,batch_size=4, num_workers=4, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val,batch_size=4, num_workers=4,)
			

