import torch
from torch.nn import functional as F
from .unet import UNet
from pytorch_lightning.core.lightning import LightningModule
from defects_dlmbl.segment_affinities import mutex_watershed
import numpy as np
from skimage.io import imsave
from inferno.extensions.criteria import set_similarity_measures as sim
from cremi_tools.metrics import cremi_metrics


class UNetModule(LightningModule):
	def __init__(self, num_fmaps=18, inc_factors=3, depth = 4, offsets=[[-1,0],[0,-1], [-9, 0], [0, -9]], separating_channel=2):
		super().__init__()
		self.offsets = offsets
		self.separating_channel=separating_channel
		self.unet = UNet(in_channels=1,
           num_fmaps=num_fmaps,
           fmap_inc_factors=inc_factors,
           downsample_factors=[[2,2] for _ in range(depth-1)],
           padding='valid')
		self.final_conv=torch.nn.Conv2d(num_fmaps,len(self.offsets), 1)
		if not offsets:
			self.offsets = [[-1,0],[0,-1]]
		else:
			self.offsets = offsets
		self.separating_channel=separating_channel
		self.DiceLoss = sim.SorensenDiceLoss()

	def forward(self,x):
		x= self.unet(x)
		x=self.final_conv(x)
		return x

	def training_step(self,batch,batch_idx):
		x,y,gt_seg=batch
		logits=self(x)
		crop_val = (y.shape[-1]-logits.shape[-1])/2
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		y = y[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_seg = gt_seg[:,:,crop_val:-crop_val,crop_val:-crop_val]
		x = x[:,:,crop_val:-crop_val,crop_val:-crop_val]
		y = y.float()
		logits *= (y!=-1).float() # ignore label -1
		
		# SDL input shape expects [b, c, ...]
		py = F.sigmoid(logits) 
		loss = self.DiceLoss(py, y)
		loss = loss+len(self.offsets)
		
		
		#loss=F.binary_cross_entropy_with_logits(logits,y)
		
		logger = self.logger.experiment
		self.log('train_loss',loss)
		if self.global_step % 100 == 0:
			
			logger.add_image('image', x[0], self.global_step)

			affinity_image = torch.sigmoid(logits)
			logger.add_image('affinity', affinity_image[0], self.global_step)
			affinity_image = affinity_image.cpu().detach().numpy()
			segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)

			logger.add_image('segmentation', segmentation[0], self.global_step, dataformats='CHW')
			logger.add_image('GT',y[0], self.global_step)
			if self.global_step % 1000 == 0:
				imsave(f'images_2/{self.global_step}_segmentation.tif', segmentation.astype(np.uint16))
				imsave(f'images_2/{self.global_step}_affinity.tif', affinity_image)
				imsave(f'images_2/{self.global_step}_gt.tif', gt_seg[0].cpu().detach().numpy())
				imsave(f'images_2/{self.global_step}_image.tif', x[0].cpu().detach().numpy())
			scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().detach().numpy())
			self.log("performance",scores)
		return loss
		
	def validation_step(self,batch,batch_idx):
		x,y,gt_seg=batch
		logits=self(x)
		crop_val = (y.shape[-1]-logits.shape[-1])/2
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		y = y[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_seg = gt_seg[:,:,crop_val:-crop_val,crop_val:-crop_val]
		y = y.float()
		logits *= (y!=-1).float() # ignore label -1
		
		# SDL input shape expects [b, c, ...]
		py = F.sigmoid(logits) 
		val_loss = self.DiceLoss(py,y)
		val_loss = val_loss+len(self.offsets)
		affinity_image = torch.sigmoid(logits).cpu().detach().numpy()	
		segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)
		print(segmentation.shape, gt_seg.shape)
		val_scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().numpy())
		self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
		self.log("val_performance", val_scores)
		return val_loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-4)