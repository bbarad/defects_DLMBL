from defects_dlmbl.unetmodule import UNetModuleWithBCEAux
from defects_dlmbl.datamodules import WingDataModule
import pytorch_lightning as pl
import argparse
from imgaug import augmenters as iaa
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser.add_argument('--test_file', type=str)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()


#### EXPERIMENT BLOCK 1 ####
img_dir = 'Wing_2'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

augmenter = iaa.Sequential([
                                    iaa.Flipud(0.5),
                                    iaa.Fliplr(0.5),
                                    iaa.LinearContrast((0.75, 1.5)),
                                    iaa.geometric.ElasticTransformation(alpha=(0, 20), sigma=10),
                                    iaa.GaussianBlur(sigma=(0, 1)),
                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 10)),
                                    iaa.Affine(scale=(0.8, 1.2), rotate=(-40, 40)),
                                    # iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.1, 0.5)),
                                    ],random_order=False)

offsets = [[-1,0],[0,-1],[-5,0],[0,-5]]
separating_channel = 2 # separates affinity from repulsion

checkpoint_callback = ModelCheckpoint(
        every_n_epochs=50,
        filename='sample-mnist-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

dm = WingDataModule(args.train_file, augmenter=augmenter, offsets = offsets)
model = UNetModuleWithBCEAux(offsets=offsets, image_dir=img_dir)
logger = TensorBoardLogger("logs", name="base_model")
trainer = pl.Trainer.from_argparse_args(args,callbacks=[checkpoint_callback])

trainer.fit(model, dm)

