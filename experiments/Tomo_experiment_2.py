from defects_dlmbl.unetmodule import UNetModule, UNetModuleSemanticWithDistance
from defects_dlmbl.datamodules import TomoDataModule, TomoDataSemModule
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


#### EXPERIMENT BLOCK 3 ####
img_dir = 'tomo_images_block_3'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
ia.seed(np.random.randint(1, 100000))
augmenter = iaa.Sequential([        iaa.Flipud(0.5),
                                    iaa.Fliplr(0.5),
                                    iaa.LinearContrast((0.75, 1.5)),
                                    iaa.geometric.ElasticTransformation(alpha=(0, 40), sigma=10),
                                    iaa.Affine(scale=(0.8, 1.2), rotate=(-15, 15)),
                                    ],random_order=False)

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='tomo-semantic-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

dm = TomoDataSemModule(args.train_file, augmenter=augmenter)
model = UNetModuleSemanticWithDistance(image_dir=img_dir)
logger = TensorBoardLogger("logs", name="base_model")
trainer = pl.Trainer.from_argparse_args(args,callbacks=[checkpoint_callback])
trainer.fit(model, dm)


