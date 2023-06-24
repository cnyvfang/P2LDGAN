import argparse
import gc
import time
import datetime
import sys

from torchvision.utils import save_image
import torchvision.models as models

from adabelief_pytorch import AdaBelief


from torch.utils.data import DataLoader
import itertools

from torch.autograd import Variable

from model.models_resnet import *


from datasets import *
from tester_datasets import *

from model.blocks import *

import torch
from logsave import *
from pytorch_fid import fid_score, inception

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img2ld_2", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")


parser.add_argument(
    "--sample_interval", type=int, default=5, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

logger = get_logger('saved_log/exp.log')


cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_GAN = 10

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
print(patch)


input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
generator = DenseSumResNeXt()
discriminator = Discriminator_patch(input_shape)


best_score = float("inf")
best_ssim = 0
best_epoch = 0

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()


if opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cuda:0'))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_patch_%d.pth" % (opt.dataset_name, opt.epoch),map_location='cuda:0'))
else:
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = AdaBelief(generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2),eps=1e-16,rectify=False,weight_decay=0)
optimizer_D = AdaBelief(discriminator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2),eps=1e-16,rectify=False,weight_decay=0)

transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    ImageDatasetX("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images():
    """Saves a generated sample from the validation set"""
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            real_A = Variable(batch["B"].type(Tensor))
            fake_B = generator(real_A)
            img_sample = fake_B.data
            save_image(img_sample, "images/%s/%d.png" % (opt.dataset_name, i), nrow=1, normalize=True)
            gc.collect()


# ----------
#  Training
# ----------

prev_time = time.time()

logger.info('start training!')

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False) #label smooth
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)


        fake_B_pred = discriminator(fake_B)
        real_B_pred = discriminator(real_B).detach()

        loss_GAN = criterion_GAN(fake_B_pred - real_B_pred.mean(0,keepdim=True), valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_pred = discriminator(real_B)
        fake_pred = discriminator(fake_B.detach())

        loss_real = criterion_GAN(real_pred - fake_pred.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(fake_pred - real_pred.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log

        logger.info('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s'%(epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,))


        gc.collect()


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:

        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))



logger.info('finish training!')
