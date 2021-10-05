from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datamodules import CREMIDataModule, TomoDataModule
import pytorch_lightning as pl
import argparse
from imgaug import augmenters as iaa
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import imgaug as ia
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()


#### EXPERIMENT BLOCK 1 ####
img_dir = 'tomo_images_block_1'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
ia.seed(np.random.randint(1, 100000))
# augmenter = iaa.Sequential([
#                                     iaa.Flipud(0.5),
#                                     iaa.Fliplr(0.5),
#                                     iaa.LinearContrast((0.75, 1.5)),
#                                     iaa.geometric.ElasticTransformation(alpha=(0, 40), sigma=10),
#                                     iaa.GaussianBlur(sigma=(0, 2)),
#                                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 20)),
#                                     iaa.Cutout(nb_iterations=(2,10), size=(.05,.2),squared=False, fill_mode='gaussian'),
#                                     iaa.Affine(scale=(0.8, 1.2), rotate=(-15, 15)),
#                                     ],random_order=False)


offsets = [[-1,0],[0,-1],[-1,-1],[-3,0],[-2,-1],[-1,-2],[0,-3],[1,-2],[2,-1]]
separating_channel = 3 # separates affinity from repulsion

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='tomo-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

dm = TomoDataModule(args.train_file, augmenter=augmenter, offsets = offsets)
model = UNetModule(offsets=offsets, image_dir=img_dir, separating_channel=separating_channel)
# logger = TensorBoardLogger("logs", name="base_model")
trainer = pl.Trainer.from_argparse_args(args,callbacks=[checkpoint_callback])

trainer.fit(model, dm)


