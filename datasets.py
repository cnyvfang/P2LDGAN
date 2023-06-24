import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image

import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.filesA = sorted(glob.glob(os.path.join(root, mode) + "/A/*.*"))
        self.filesB = sorted(glob.glob(os.path.join(root, mode) + "/B/*.*"))

        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "val") + "/*.*")))

    def augmentation(self, img_B, img_A):
        # Data augmentation
        # Random horizontal flip
        if np.random.random() < 0.2:
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")

        # Random vertical flip
        if np.random.random() < 0.2:
            img_B = Image.fromarray(np.array(img_B)[::-1, :, :], "RGB")
            img_A = Image.fromarray(np.array(img_A)[::-1, :, :], "RGB")

        # Random rotation
        if np.random.random() < 0.2:
            angle = random.randint(-180, 180)
            fff = Image.new('RGBA', img_A.size, (255,) * 4)
            img_A = img_A.convert('RGBA')
            img_B = img_B.convert('RGBA')
            img_B = img_B.rotate(angle,expand=1)
            img_A = img_A.rotate(angle,expand=1)
            img_A = Image.composite(img_A, fff, img_A)
            img_B = Image.composite(img_B, fff, img_B)
            img_A = img_A.convert('RGB')
            img_B = img_B.convert('RGB')

        #Random scaling
        if np.random.random() < 0.2:
            scale = random.uniform(0.9, 1.5)
            w, h = img_B.size
            fff = Image.new('RGBA', img_A.size, (255,) * 4)
            img_B = img_B.resize((int(img_B.size[0] * scale), int(img_B.size[1] * scale)), Image.BICUBIC)
            img_A = img_A.resize((int(img_A.size[0] * scale), int(img_A.size[1] * scale)), Image.BICUBIC)
            #Image cropping back to original resolution
            if scale >1:
                img_B = img_B.crop((0, 0, w, h))
                img_A = img_A.crop((0, 0, w, h))
            else:
                img_A = img_A.convert('RGBA')
                img_B = img_B.convert('RGBA')
                img_A = Image.composite(img_A, fff, img_A)
                img_B = Image.composite(img_B, fff, img_B)
                img_A = img_A.convert('RGB')
                img_B = img_B.convert('RGB')


        # Random crop
        if np.random.random() < 0.2:
            w, h = img_B.size
            x = random.randint(512, 768)
            x1 = random.randint(0, w - x)
            y1 = random.randint(0, h - x)
            img_B = img_B.crop((x1, y1, x1 + x, y1 + x))
            img_A = img_A.crop((x1, y1, x1 + x, y1 + x))

        # Random resize
        elif np.random.random() < 0.2:
            w, h = img_B.size
            fff = Image.new('RGBA', img_A.size, (255,) * 4)
            #random Distortion and deformation
            r1 = random.randint(-500, 0)
            r2 = random.randint(-500, 0)
            img_B = transforms.Resize((h + r1, w + r2))(img_B)
            img_A = transforms.Resize((h + r1, w + r2))(img_A)
            img_A = img_A.convert('RGBA')
            img_B = img_B.convert('RGBA')
            img_A = Image.composite(img_A, fff, img_A)
            img_B = Image.composite(img_B, fff, img_B)
            img_A = img_A.convert('RGB')
            img_B = img_B.convert('RGB')

        else:
            None

        # Random brightness
        if np.random.random() < 0.2:
            r = random.uniform(0.7, 1.3)
            img_B = transforms.ColorJitter(brightness=r)(img_B)

        # Random contrast
        if np.random.random() < 0.2:
            r = random.uniform(0.7, 1.3)
            img_B = transforms.ColorJitter(contrast=r)(img_B)

        # Random saturation 饱和度
        if np.random.random() < 0.2:
            r = random.uniform(0.7, 1.3)
            img_B = transforms.ColorJitter(saturation=r)(img_B)

        # Random hue 色相
        if np.random.random() < 0.2:
            r = random.uniform(0, 0.5)
            img_B = transforms.ColorJitter(hue=r)(img_B)

        # Random blur
        if np.random.random() < 0.2:
            img_B = transforms.GaussianBlur(1)(img_B)

        #Converting images to greyscale
        if np.random.random() < 0.2:
            img_B = transforms.Grayscale(num_output_channels=3)(img_B)


        return img_B, img_A


    def __getitem__(self, index):

        img_B = Image.open(self.filesA[index % len(self.filesA)]) #original image
        img_A = Image.open(self.filesB[index % len(self.filesB)]) #target image

        img_B, img_A = self.augmentation(img_B, img_A)

        img_B = self.transform(img_B)
        img_A = self.transform(img_A)

        return {"A": img_A, "B": img_B} #target image, opiginal image

    def get_all(self):
        img_B = Image.open(self.filesA) #original image
        img_A = Image.open(self.filesB) #target image

        img_B = self.transform(img_B)
        img_A = self.transform(img_A)

        return {"A": img_A, "B": img_B} #target image, opiginal image

    def __len__(self):
        return len(self.filesA)
