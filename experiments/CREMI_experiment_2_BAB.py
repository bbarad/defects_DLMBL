from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datamodules import CREMIDataModule
import pytorch_lightning as pl
import argparse
from imgaug import augmenters as iaa
from pytorch_lightning.loggers import TensorBoardLogger
import os



#### EXPERIMENT BLOCK 1 ####
img_dir = 'images_block_1'
if not os.path.exists(img_dir)
    os.makedirs(img_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
augmenter = iaa.Sequential([iaa.Affine(scale=(0.8, 1.2), rotate=(-25, 25)),
                                    iaa.Flipud(0.5),
                                    iaa.Fliplr(0.5),
                                    iaa.LinearContrast((0.75, 1.5)),
                                    iaa.geometric.ElasticTransformation(alpha=(0, 20), sigma=10),
                                    iaa.GaussianBlur(sigma=(0, 1)),
                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 10)),
                                    # iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.1, 0.5)),
                                    ],random_order=False)

offsets = [[-1,0],[0,-1],[-9,0],[0,-9]]
separating_channel = 2 # separates affinity from repulsion

dm = CREMIDataModule(args.train_file, augmenter=augmenter, offsets = offsets)
model = UNetModule(offsets=offsets, image_dir=img_dir)
logger = TensorBoardLogger("logs", name="base_model")
trainer = pl.Trainer.from_argparse_args(args)


trainer.fit(model, dm)