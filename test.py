import argparse
import gc
import os.path
import sys

from torchvision.utils import save_image
from model.models import *

from torch.autograd import Variable
from PIL import Image

import numpy as np

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--img_height", type=int, default=1024, help="size of image height")
parser.add_argument("--img_width", type=int, default=1024, help="size of image width")
parser.add_argument("--image", type=str, default='./image/RM.jpg', help="the image used to translate")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--post_processing", type=bool, default=True, help="The results are enhanced by an iterative generation method. (This method was not used in the thesis)")

opt = parser.parse_args()
print(opt)

# torch.cuda.set_d sevice(1)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


cuda = True if torch.cuda.is_available() else False
best_score = float("inf")  # best val accuracy
best_ssim = 0

generator = Generator()

if cuda:
    generator = generator.cuda()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Loading model")
generator.load_state_dict(torch.load("./generator.pth"))
generator.eval()

print("Translating")
img_A = Image.open(opt.image) #target image
img_A = img_A.convert("RGB")
img_A = transforms.CenterCrop(img_A.size[1])(img_A)
img_A = transforms.Resize((1024),Image.BICUBIC)(img_A)

# img_A = transforms.ColorJitter(contrast=1.5)(img_A)
# img_A = transforms.Resize((int(img_A.size[1]/2),int(img_A.size[1]/2)),Image.BICUBIC)(img_A)

img_A = transforms.ToTensor()(img_A)
img_A = img_A.unsqueeze(0)
img_A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_A)
img_A = Variable(img_A.type(Tensor))

print(img_A.shape)

with torch.no_grad():

    fake_B = generator(img_A)
    fake_B_prev = fake_B.data

    if opt.post_processing == True:
        for _ in range(2):
            combine = (fake_B/5) + img_A
            fake_B = generator(combine)
            # fake_B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fake_B)
            print("Deep Sampling")

    sample = fake_B.data
    save_image(sample, "image/output.png")
    save_image(fake_B_prev, "image/output_o.png")
print("Done.")
gc.collect()

