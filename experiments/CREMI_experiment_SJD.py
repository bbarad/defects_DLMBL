from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datamodules import CREMIDataModule
import pytorch_lightning as pl
import argparse
from imgaug import augmenters as iaa
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

augmenter = iaa.Sequential([iaa.Affine(scale=(0.8, 1.2), rotate=(-25, 25)),
                                    iaa.Flipud(0.5),
                                    iaa.Fliplr(0.5),
                                    iaa.LinearContrast((0.75, 1.5)),
                                    iaa.geometric.ElasticTransformation(alpha=(0, 20), sigma=10),
                                    iaa.GaussianBlur(sigma=(0, .75)),
                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, .5)),
                                    iaa.arithmetic.ReplaceElementwise((0, .05), (.05, .5))
                                    ],random_order=False)

dm = CREMIDataModule(args.train_file, augmenter=augmenter)
model = UNetModule()
logger = TensorBoardLogger("logs", name="base_model")
trainer = pl.Trainer.from_argparse_args(args)


trainer.fit(model, dm)