from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .datasets import ISBIDataset, CREMIDataset

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
	def __init__(self, train_filename, augmenter = None, augment_and_crop=False, pad=0):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename
		self.augmenter = augmenter
		self.augment_and_crop = augment_and_crop
		self.pad = pad


	def setup(self, stage = None):
		full_dataset = len(CREMIDataset(self.train_filename))
		split_index = 4*full_dataset//5
		train_indices = list(range(split_index))
		val_indices = list(range(split_index,full_dataset))
		self.train = CREMIDataset(self.train_filename,indices=train_indices, augmenter=self.augmenter)
		self.val = CREMIDataset(self.train_filename,indices=val_indices)	

	def train_dataloader(self):
		return DataLoader(self.train,batch_size=1, num_workers=8, augment_and_crop=self.augment_and_crop, pad=self.pad)

	def val_dataloader(self):
		return DataLoader(self.val,batch_size=1, num_workers=8, augment_and_crop=self.augment_and_crop, pad=self.pad)