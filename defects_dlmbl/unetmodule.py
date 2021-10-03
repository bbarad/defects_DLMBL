import torch
from torch.nn import functional as F
from .unet import UNet
from pytorch_lightning.core.lightning import LightningModule
from defects_dlmbl.segment_affinities import mutex_watershed
import numpy as np
from skimage.io import imsave

from cremi_tools.metrics import cremi_metrics


class UNetModule(LightningModule):
	def __init__(self, num_fmaps=18, num_affinities=4, inc_factors=3, depth = 4, offsets=None, separating_channel=2):
		super().__init__()
		self.unet = UNet(in_channels=1,
           num_fmaps=num_fmaps,
           fmap_inc_factors=inc_factors,
           downsample_factors=[[2,2] for _ in range(depth)],
           padding='same')
		self.final_conv=torch.nn.Conv2d(num_fmaps,num_affinities, 1)
		if not offsets:
			self.offsets = [[-1,0],[0,-1]]
		else:
			self.offsets = offsets
		self.separating_channel=separating_channel

	def forward(self,x):
		x= self.unet(x)
		x=self.final_conv(x)
		return x

	def training_step(self,batch,batch_idx):
		x,y,gt_seg=batch
		logits=self(x)
		y = y.float()
		logits *= (y!=-1).float() # ignore label -1
		loss=F.binary_cross_entropy_with_logits(logits,y)
		logger = self.logger.experiment
		self.log('train_loss',loss)
		if self.global_step % 100 == 0:
			
			logger.add_image('image', x.squeeze(0))

			affinity_image = torch.sigmoid(logits).squeeze(0)
			logger.add_image('affinity', affinity_image)
			affinity_image = affinity_image.squeeze(0).cpu().detach().numpy()

			# affs = np.stack([
			# np.zeros_like(affinity_image[0]),
			# affinity_image[0],
			# affinity_image[1]]
			# )
			# waterz agglomerate requires 4d affs (c, d, h, w) - add fake z dim
			# affs = np.expand_dims(affs, axis=1)
			# segmentation = seg.watershed_from_affinities(affs, threshold=0.95)
			segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)
			logger.add_image('segmentation', segmentation,dataformats='HW')
			logger.add_image('GT',y.squeeze(0))
			if self.global_step % 1000 == 0:
				imsave(f'images/{self.global_step}_segmentation.tif', segmentation.astype(np.uint16))
				imsave(f'images/{self.global_step}_affinity.tif', affinity_image)
				imsave(f'images/{self.global_step}_gt.tif', gt_seg.cpu().detach().numpy().squeeze(0))
				imsave(f'images/{self.global_step}_image.tif', x.cpu().detach().numpy().squeeze(0))

			scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().detach().numpy().squeeze(0))
			self.log("performance", scores)
		return loss
		
	def validation_step(self,batch,batch_idx):
		x,y,gt_seg=batch
		logits=self(x)
		y = y.float()
		logits *= (y!=-1).float()
		affinity_image = torch.sigmoid(logits).squeeze(0).cpu().detach().numpy()
		# affs = np.stack([
		# 	np.zeros_like(affinity_image[0]),
		# 	affinity_image[0],
		# 	affinity_image[1]]
		# )
		# waterz agglomerate requires 4d affs (c, d, h, w) - add fake z dim
		# affs = np.expand_dims(affs, axis=1)
		# segmentation = seg.watershed_from_affinities(affs, threshold=0.95)
		segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)
		val_loss=F.binary_cross_entropy_with_logits(logits,y)
		val_scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().numpy().squeeze())
		self.log("val_loss", val_loss)
		self.log("val_performance", val_scores)



	
	
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-4)