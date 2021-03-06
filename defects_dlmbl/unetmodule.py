import torch
from .disc_loss import DiscriminativeLoss
from .unet import UNet
from pytorch_lightning.core.lightning import LightningModule
from defects_dlmbl.segment_affinities import mutex_watershed
import numpy as np
from skimage.io import imsave
import io
import PIL
from sklearn.decomposition import PCA
from inferno.extensions.criteria import set_similarity_measures as sim
from cremi_tools.metrics import cremi_metrics
import matplotlib.pyplot as plt
import torch.nn.functional as F


class UNetModule(LightningModule):
	def __init__(self, num_fmaps=18, inc_factors=3, depth = 4, offsets=[[-1,0],[0,-1], [-9, 0], [0, -9]], separating_channel=2, image_dir="images"):
		super().__init__()
		self.num_fmaps=num_fmaps
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
		self.image_dir = image_dir

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
		py = torch.sigmoid(logits) 
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
				imsave(f'{self.image_dir}/{self.global_step}_segmentation.tif', segmentation.astype(np.uint16))
				imsave(f'{self.image_dir}/{self.global_step}_affinity.tif', affinity_image)
				imsave(f'{self.image_dir}/{self.global_step}_gt.tif', gt_seg[0].cpu().detach().numpy())
				imsave(f'{self.image_dir}/{self.global_step}_image.tif', x[0].cpu().detach().numpy())
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
		py = torch.sigmoid(logits) 
		val_loss = self.DiceLoss(py,y)
		val_loss = val_loss+len(self.offsets)
		affinity_image = torch.sigmoid(logits).cpu().detach().numpy()	
		segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)
		val_scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().numpy())
		self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
		self.log("val_performance", val_scores)
		return val_loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-4)

class UNetModuleWithMetricAuxiliary(UNetModule):
	def __init__(self,metric_dimensions=16,loss_alpha=0.5,**kwargs):
		super().__init__(**kwargs)
		self.metric_dimensions = metric_dimensions
		self.loss_alpha = loss_alpha
		self.output_dims = len(self.offsets)+self.metric_dimensions
		self.final_conv=torch.nn.Conv2d(self.num_fmaps,self.output_dims, 1)
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.DiscriminativeLoss = DiscriminativeLoss(device)
		self.pca = PCA()

	def calculate_loss(self,logits_aff,logits_metric,gt_aff,gt_seg):
		# affinity loss
		py = torch.sigmoid(logits_aff) 
		loss_aff = self.DiceLoss(py, gt_aff)
		loss_aff += len(self.offsets)
		loss_aff /= len(self.offsets)
		# metric loss
		loss_metric = self.DiscriminativeLoss(logits_metric,gt_seg)
		return (self.loss_alpha)*loss_aff + (1-self.loss_alpha)*loss_metric

	def scatter_metric_pca(self,logits_metric,sample=3):
		# making pca scatter if not doing pixel-wise pca vizualization
		sampled = logits_metric[...,::sample,::sample].reshape(self.metric_dimensions,-1)
		sampled_pca = self.pca.fit_transform(sampled.T)
		fig,ax = plt.subplots(1,1,figsize=(10,10))
		ax.scatter(*sampled_pca.T[:2])
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.close(fig)
		p = PIL.Image.open(buf)
		return torch.tensor(np.array(p))

	def scatter_metric_pca_from_image(self,image_metric_pca,sample=3):
		# sample from pixel-wize pca image and display first two dimensions as scatter plot
		sampled_pca = image_metric_pca[:2,::sample,::sample]
		fig,ax = plt.subplots(1,1,figsize=(10,10))
		ax.scatter(*sampled_pca[:2])
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.close(fig)
		p = PIL.Image.open(buf)
		return torch.tensor(np.array(p))

	def image_metric_pca(self,logits_metric,return_dimensions=3):
		full_pca = self.pca.fit_transform(logits_metric.reshape(self.metric_dimensions,-1).T)
		full_pca = full_pca.T.reshape(logits_metric.shape)[:return_dimensions]
		full_pca -= full_pca.min(axis=(-2,-1),keepdims=True)
		full_pca /= full_pca.max(axis=(-2,-1),keepdims=True)
		return full_pca

	def training_step(self,batch,batch_idx):
		x,gt_aff,gt_seg=batch
		logits=self(x)
		crop_val = (gt_aff.shape[-1]-logits.shape[-1])/2
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		gt_aff = gt_aff[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_seg = gt_seg[:,:,crop_val:-crop_val,crop_val:-crop_val]
		x = x[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_aff = gt_aff.float()

		logits_aff = logits[:,:len(self.offsets)]
		logits_metric = logits[:,len(self.offsets):]

		assert self.metric_dimensions == logits_metric.shape[1], "logits_metric channels do not match metric_dimensions"

		logits_aff *= (gt_aff!=-1).float() # ignore label -1

		loss = self.calculate_loss(logits_aff,logits_metric,gt_aff,gt_seg)
		
		logger = self.logger.experiment
		self.log('train_loss',loss)
		if self.global_step % 1000 == 0:
			
			logger.add_image('image', x[0], self.global_step)

			affinity_image = torch.sigmoid(logits_aff)
			logger.add_image('affinity', affinity_image[0], self.global_step)
			affinity_image = affinity_image.cpu().detach().numpy()
			segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)

			logger.add_image('segmentation', segmentation[0], self.global_step, dataformats='CHW')
			logits_metric_viz = logits_metric[0].detach().cpu().numpy()
			logits_metric_viz -= logits_metric_viz.min(axis=(-2,-1),keepdims=True)
			logits_metric_viz /= logits_metric_viz.max(axis=(-2,-1),keepdims=True)
			image_metric_pca = self.image_metric_pca(logits_metric_viz)
			logger.add_image('metric_scatter',self.scatter_metric_pca_from_image(image_metric_pca), self.global_step,dataformats='HWC')
			logger.add_image('metric_image',image_metric_pca,self.global_step)
			logger.add_image('GT',gt_aff[0], self.global_step)
			if self.global_step % 100 == 0:
				imsave(f'{self.image_dir}/{self.global_step}_segmentation.tif', segmentation.astype(np.uint16))
				imsave(f'{self.image_dir}/{self.global_step}_affinity.tif', affinity_image)
				imsave(f'{self.image_dir}/{self.global_step}_gt.tif', gt_seg[0].cpu().detach().numpy())
				imsave(f'{self.image_dir}/{self.global_step}_image.tif', x[0].cpu().detach().numpy())
			scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().detach().numpy())
			self.log("performance",scores)
		return loss
		
	def validation_step(self,batch,batch_idx):
		x,gt_aff,gt_seg=batch
		logits=self(x)
		crop_val = (gt_aff.shape[-1]-logits.shape[-1])/2
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		gt_aff = gt_aff[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_seg = gt_seg[:,:,crop_val:-crop_val,crop_val:-crop_val]
		gt_aff = gt_aff.float()

		logits_aff = logits[:,:len(self.offsets)]
		logits_metric = logits[:,len(self.offsets):]

		logits_aff *= (gt_aff!=-1).float() # ignore label -1
		
		val_loss = self.calculate_loss(logits_aff,logits_metric,gt_aff,gt_seg)

		affinity_image = torch.sigmoid(logits_aff).cpu().detach().numpy()
		segmentation = mutex_watershed(affinity_image,self.offsets,self.separating_channel,strides=None)
		val_scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().numpy())
		self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
		self.log("val_performance", val_scores)
		return val_loss

class UNetModuleSemanticWithDistance(UNetModule):
	"""Same UNet, but predict 2 layers instead of a million"""
	def __init__(self, num_fmaps=32, inc_factors=2, depth = 4, image_dir="images"):
		super().__init__()
		self.num_fmaps=num_fmaps
		self.unet = UNet(in_channels=1,
           num_fmaps=num_fmaps,
           fmap_inc_factors=inc_factors,
           downsample_factors=[[2,2] for _ in range(depth-1)],
           padding='valid')
		self.final_conv=torch.nn.Conv2d(num_fmaps, 2, 1)
		self.L1loss = torch.nn.L1Loss()
		self.DiceLoss = sim.SorensenDiceLoss()
		self.loss_alpha = 0.5
		self.tanh = torch.nn.Tanh()
		self.image_dir = image_dir
		
	
	def calculate_loss(self,seg,dist,gt_seg,gt_dist):
		# affinity loss
		 
		loss_seg = self.DiceLoss(seg, gt_seg)
		# metric loss
		loss_dist = self.L1loss(dist,gt_dist)
		return (self.loss_alpha)*loss_seg + (1-self.loss_alpha)*loss_dist

	def training_step(self,batch,batch_idx):
		x, y = batch # batch size > 1
		logits = self(x)
		crop_val = (y.shape[-1]-logits.shape[-1])/2 # only works if square
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		y = y[:,:,crop_val:-crop_val,crop_val:-crop_val]
		x = x[:,:,crop_val:-crop_val,crop_val:-crop_val]

		logits_seg = logits[:,:1]
		seg = F.sigmoid(logits_seg)
		
		logits_dist = logits[:,1:]
		# dist = self.tanh(logits_dist)
		dist = torch.clamp(logits_dist, -30, 30)
		dist_seg = dist < 0

		gt_seg = y[:,:1]
		gt_dist = y[:,1:]
		loss = self.calculate_loss(logits_seg,logits_dist,gt_seg,gt_dist)

		if self.global_step % 100 == 0:
			logger = self.logger.experiment
			logger.add_image('image', x[0], self.global_step)
			logger.add_image('segmentation', seg[0], self.global_step)
			logger.add_image('Distance based segmentation', dist_seg[0], self.global_step)
			logger.add_image('GT seg',gt_seg[0], self.global_step)
			logger.add_image('distance',dist[0], self.global_step)
			logger.add_image('GT dist',gt_dist[0], self.global_step)
			imsave(f'{self.image_dir}/{self.global_step}_learnedsegmentation.tif', seg.cpu().detach().numpy())
			imsave(f'{self.image_dir}/{self.global_step}_distance.tif', dist.cpu().detach().numpy())
			imsave(f'{self.image_dir}/{self.global_step}_distance_seg.tif', dist_seg.cpu().detach().numpy())
			imsave(f'{self.image_dir}/{self.global_step}_gt.tif', gt_seg.cpu().detach().numpy())
			imsave(f'{self.image_dir}/{self.global_step}_image.tif', x.cpu().detach().numpy())

		self.log('train_loss',loss)
		try:
			scores = cremi_metrics.cremi_scores(dist_seg.cpu().detach().numpy(), gt_seg.cpu().detach().numpy())
			self.log("performance",scores)
		except:
			pass
		return loss

	def validation_step(self,batch,batch_idx):
		x, y = batch
		logits = self(x)
		crop_val = (y.shape[-1]-logits.shape[-1])/2 # only works if square
		assert crop_val == int(crop_val), "Can't crop by an odd total pixel count"
		crop_val = int(crop_val)
		y = y[:,:,crop_val:-crop_val,crop_val:-crop_val]
		x = x[:,:,crop_val:-crop_val,crop_val:-crop_val]
		logits_seg = logits[:,:1]
		seg = F.sigmoid(logits_seg)
		
		logits_dist = logits[:,1:]
		dist =torch.clamp(logits_dist, -30, 30)
		# dist = self.tanh(logits_dist)
		gt_seg = y[:,:1]
		gt_dist = y[:,1:]
		val_loss = self.calculate_loss(seg,dist,gt_seg,gt_dist)
		try:
			val_scores = cremi_metrics.cremi_scores(segmentation, gt_seg.cpu().numpy())
			self.log("val_performance", val_scores)
		except:
			pass
		self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
		return val_loss
