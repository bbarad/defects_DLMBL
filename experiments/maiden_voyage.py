from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datamodules import ISBIDataModule
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

dm = ISBIDataModule(args.train_file)
model = UNetModule()
trainer = pl.Trainer.from_argparse_args(args)
trainer.fit(model, dm)