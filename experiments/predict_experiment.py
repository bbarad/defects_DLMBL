from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datasets import CREMIDataset
from defects_dlmbl.segment_affinities import mutex_watershed
import torch
import pytorch_lightning as pl
import argparse

model = UNetModule.load_from_checkpoint(checkpoint_path="lightning_logs/version_112/checkpoints/epoch=999-step=98999.ckpt")

model.eval()
with torch.no_grad():
    dataset = CREMIDataset("../data/training_data.zarr", pad=93, augmenter=None, augment_and_crop=False)
    for i in range(dataset.get_num_samples()):
        x, aff, y = dataset[i]
        x = x.unsqueeze(0).unsqueeze(0)
        model = model.cuda()
        x = x.cuda()
        aff_pred = model(x)
        aff_pred = aff_pred[...,1:-1,1:-1]
        segmentation = mutex_watershed(aff_pred, [(1,0),(0,1)], 1)
        print(segmentation.shape)

        
