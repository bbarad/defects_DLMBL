import torch
from torch.nn import functional as F
from .unet import UNet
from pytorch_lightning.core.lightning import LightningModule
import segment_affinities as seg
import numpy as np

class UNetModule(LightningModule):
	def __init__(self, num_fmaps=12, num_affinities=2):
		super().__init__()
		self.unet = UNet(in_channels=1,
           num_fmaps=num_fmaps,
           fmap_inc_factors=3,
           downsample_factors=[[2,2],[2,2],[2,2]],
           padding='same')
		self.final_conv=torch.nn.Conv2d(num_fmaps,num_affinities, 1)

	def forward(self,x):
		x= self.unet(x)
		x=self.final_conv(x)
		return x

	def training_step(self,batch,batch_idx):
		x,y=batch
		logits=self(x)
		y = y.float()
		logits *= (y!=-1).float() # ignore label -1
		loss=F.binary_cross_entropy_with_logits(logits,y)
		logger = self.logger.experiment
		self.log('train_loss',loss)
		if self.global_step % 100 == 0:
			
			logger.add_image('image', x.squeeze(0))

			affinity_image = torch.sigmoid(logits)
			logger.add_image('affinity', affinity_image.squeeze(0))
			affs = np.stack([
			np.zeros_like(affs[0]),
			affinity_image[0],
			affinity_image[1]]
			)
			# waterz agglomerate requires 4d affs (c, d, h, w) - add fake z dim
			affs = np.expand_dims(affs, axis=1)
			segmentation = seg.watershed_from_affinities(affs)
			logger.add_image('segmentation', segmentation[0].squeeze())
			logger.add_image('GT',y.squeeze(0))
		return loss
	
	
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-4)