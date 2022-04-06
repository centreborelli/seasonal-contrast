import pathlib

import numpy as np
import torch
from torchvision.transforms import functional as TF

from datasets import oscd_dataset
from main_oscd import SiamSegment


# load model
model = SiamSegment.load_from_checkpoint("oscd_model.ckpt")

im1 = oscd_dataset.read_image(pathlib.Path("datasets/oscd/abudhabi/imgs_1"), bands=oscd_dataset.RGB_BANDS)
im2 = oscd_dataset.read_image(pathlib.Path("datasets/oscd/abudhabi/imgs_2"), bands=oscd_dataset.RGB_BANDS)

# c1 = im1.crop((0, 0, 96, 96))
# c2 = im2.crop((0, 0, 96, 96))
# x1 = TF.to_tensor(c1).unsqueeze(0)
# x2 = TF.to_tensor(c2).unsqueeze(0)
# out = model(x1, x2)
# pred = torch.sigmoid(out)

output = np.zeros((im1.height, im1.width))

for x in range(0, im1.width - 96, 96):
    for y in range(0, im1.height - 96, 96):
        c1 = im1.crop((x, y, x+96, y+96))
        c2 = im2.crop((x, y, x+96, y+96))
        out = model(TF.to_tensor(c1).unsqueeze(0),
                    TF.to_tensor(c2).unsqueeze(0))
        pred = torch.sigmoid(out)
        output[y:y+96, x:x+96] = pred.detach().numpy().squeeze()
