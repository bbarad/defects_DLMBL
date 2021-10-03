from defects_dlmbl.unetmodule import UNetModule
from defects_dlmbl.datasets import CREMIDataset
import defects_dlmbl.segment_affinities as seg
import torch
import pytorch_lightning as pl
import argparse
import numpy as np
from cremi_tools.metrics import cremi_metrics
import matplotlib.pyplot as plt

model = UNetModule.load_from_checkpoint(checkpoint_path="lightning_logs/version_112/checkpoints/epoch=999-step=98999.ckpt")

model.eval()
images = []
labels = []
pred_labels = []
with torch.no_grad():
    dataset = CREMIDataset("../data/training_data.zarr", pad=93, augmenter=None, augment_and_crop=False)
    for i in range(0,20,4):
        x, aff, y = dataset[i]
        x = x.unsqueeze(0).unsqueeze(0)
        model = model.cuda()
        x = x.cuda()
        aff_pred = model(x)
        aff_pred = aff_pred[...,1:-1,1:-1] 
        aff_pred = aff_pred.cpu().numpy()
        aff_pred = np.stack([np.zeros_like(aff_pred[0][0]),
            aff_pred[0][0],
            aff_pred[0][1]])

        aff_pred = np.expand_dims(aff_pred, axis=1)
        segmentation = seg.watershed_from_affinities(aff_pred, threshold=0.95)
        segmentation = segmentation.astype('int16')
        y = y.numpy().astype('int16')
        print(segmentation.shape, y.shape)
        print(cremi_metrics.cremi_scores(segmentation, y))
        images.append(x.squeeze(0).squeeze(0).cpu().numpy()[93:-93, 93:-93])
        labels.append(y.squeeze())
        pred_labels.append(segmentation.squeeze())
    
n_col = 5 
n_row = 3
fig, axs = plt.subplots(n_row, n_col, figsize=(15, 12))
# sample_data = [imgs[3+18*i] for i in range(5)]
sample_data = range(5)

# i = 3
for index, val in enumerate(sample_data):
    ax = axs[0, index]
    ax.imshow(images[index], cmap='gray')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax = axs[1, index]
    ax.imshow(labels[index], cmap='prism')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax = axs[2, index]
    ax.imshow(pred_labels[index], cmap='prism')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # i += 18


plt.tight_layout()
plt.show()
fig.savefig('grid.png')


        
