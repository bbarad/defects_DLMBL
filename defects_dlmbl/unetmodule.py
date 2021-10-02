import torch
from torch.nn import functional as F
from .unet import UNet
from pytorch_lightning.core.lightning import LightningModule

class UNetModule(LightningModule):
	def __init__(self):
		super().__init__()

		self.unet = UNet(in_channels=1,
           num_fmaps=6,
           fmap_inc_factors=2,
           downsample_factors=[[2,2],[2,2],[2,2]],
           padding='same',
           num_fmaps_out=2)

	def forward(self,x):
		return self.unet(x)

	def training_step(self,batch,batch_idx):
		x,y=batch
		logits=self(x)
		y = y.float()
		loss=F.binary_cross_entropy_with_logits(logits,y)
		logger = self.logger.experiment
		self.log('train_loss',loss)
		logger.add_image('image', x.squeeze(0))
		logger.add_image('affinity', logits.squeeze(0))
		logger.add_image('GT',y.squeeze(0))
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-3)