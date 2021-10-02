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
		loss=F.nll_loss(logits,y)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(),lr=1e-3)