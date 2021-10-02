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
	def __init__(self, train_filename):
		super().__init__()
		self.train_dims = None
		self.train_filename = train_filename

	def setup(self, stage = None):
		self.train = CREMIDataset(self.train_filename)

	def train_dataloader(self):
		return DataLoader(self.train,batch_size=1, num_workers=8)
	
	# def train_dataloader(self):
	# 	return DataLoader(self.test,batch_size=1)

	# def val_dataloader(self):
	# 	return DataLoader(self.val,batch_size=1)