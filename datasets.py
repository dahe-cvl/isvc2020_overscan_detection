from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
import logging
from os.path import splitext
from os import listdir
import torch
from torch.utils import data
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob
import cv2
import os
import random


class Sharp(object):
    def __call__(self, frame):
        frame = np.asarray(frame)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        frame_sharp = cv2.filter2D(frame, -1, kernel)
        frame_sharp = Image.fromarray(frame_sharp)
        return frame_sharp

    def __repr__(self):
        return self.__class__.__name__ + 'Sharp'


class Blur(object):

    def __init__(self, kernel_size=(5, 5), sigmaX=5, sigmaY=5):
        self.kernel_size = kernel_size;
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def __call__(self, frame):
        frame = np.asarray(frame)
        # print(type(frame))
        # print(frame.shape)
        frame_cv = frame.transpose(1, 2, 0)
        blurred_frame_cv = cv2.GaussianBlur(frame_cv, self.kernel_size, sigmaX=self.sigmaX, sigmaY=self.sigmaY)
        frame_final = blurred_frame_cv.transpose(2, 0, 1)  # .astype('uint8')
        return frame_final

    def __repr__(self):
        return self.__class__.__name__ + 'Blur'


class FlipPair(object):
    # Vertical flipping: 0
    # Horizontal flipping: 1
    # horizontal and vertical flipping: -1

    def __call__(self, sample):
        image, mask = sample

        image = np.asarray(image) #.transpose(1, 2, 0)
        image = np.asarray(image).transpose(1, 2, 0)

        mask = np.asarray(mask)
        mask = np.expand_dims(mask, axis=2)

        frame_dim_orig = image.shape
        label_dim_orig = mask.shape

        idx = random.randint(2, 3)

        if(idx == 0):
            flipcode = 0
            image = cv2.flip(image, flipcode)
            mask = cv2.flip(mask, flipcode)
        elif(idx == 1):
            flipcode = 1
            image = cv2.flip(image, flipcode)
            mask = cv2.flip(mask, flipcode)
        elif (idx == 2):
            flipcode = -1
            image = cv2.flip(image, flipcode)
            mask = cv2.flip(mask, flipcode)
        elif(idx == 3):
            image = image
            mask = mask

        image = cv2.resize(image, (frame_dim_orig[1], frame_dim_orig[0]))
        image = image.transpose(2, 0, 1)   #.astype('uint8')

        mask = cv2.resize(mask, (label_dim_orig[1], label_dim_orig[0]))
        mask = mask.astype('uint8')

        sample = torch.tensor(image), torch.tensor(mask)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + 'flippair'


class Normalize(object):

    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        image, mask = sample
        image_pil = F.to_pil_image(image)
        image_pil = F.to_tensor(image_pil)
        image = F.normalize(image_pil, self.mean, self.std, self.inplace)

        sample = image, mask
        return sample

    def __repr__(self):
        return self.__class__.__name__ + 'Normalize'


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
        or if the numpy.ndarray has dtype = np.uint8
        In the other cases, tensors are returned without scaling.
        """
    def __call__(self, sample):
        image, mask = sample
        sample = F.to_tensor(image), F.to_tensor(mask)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + 'ToTensor'


class ToGrayScale(object):
    def __call__(self, sample):
        image, mask = sample
        image = np.asarray(image)
        mask = np.asarray(mask)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        image_gray = image_gray.transpose(2, 0, 1)

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        sample = torch.tensor(image_gray), torch.tensor(mask_gray)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + 'ToGrayScale'


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, dim=None):
        if(dim == None):
            print("you have to specify the parameter: dim")
            exit()
        self.dim = dim

    def __call__(self, sample):
        image, mask = sample
        image = np.asarray(image)
        mask = np.asarray(mask)

        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, (self.dim[1], self.dim[0]))
        image = image.transpose(2, 0, 1)

        mask = cv2.resize(mask, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_NEAREST)

        image_pil = F.to_pil_image(image)
        image_pil = F.to_tensor(image_pil)

        sample = torch.tensor(image), torch.tensor(mask)
        return sample


class BasicDataset(data.Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1.):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        id_list = []
        for idx in self.ids:
            tmp = idx.split('_')[1]
            id_list.append(tmp)

        self.ids = id_list

        #print(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        #pil_img = pil_img.resize((newW, newH))

        pil_img = pil_img.resize((240, 180))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + "l_" + idx + '*')
        img_file = glob(self.imgs_dir + "s_" + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


def loadBasicDataset(db_path="", batch_size=8):
    dir_img = db_path + "/samples/"
    dir_mask = db_path + "/labels/"
    val_percent = 0.2
    img_scale = 1.0

    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader


class SegDataset(data.Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None,
                 data_aug_flag=False, dim=None):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_aug_flag = data_aug_flag
        self.dim = dim

        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert (subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)
            self.image_list = self.image_list[indices]
            self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        msk_name = self.mask_names[idx]
        mask = cv2.imread(msk_name)
        mask_pil = Image.fromarray(mask)

        sample = image_pil, mask_pil
        if(self.transform != None):
            sample = self.transform(sample)

        return sample


def loadSegDataset(data_dir, imageFolder='samples', maskFolder='labels', fraction=0.2, batch_size=4,
                                 expType="gray", data_aug_flag=False, dim=None):

    if (expType == "gray"):
        data_transforms = {
            'Train': transforms.Compose([
                ToGrayScale(),
                Resize(dim),
                FlipPair(),
                Normalize(mean=[110.0057 / 255.0,
                                110.0057 / 255.0,
                                110.0057 / 255.0],
                          std=[62.1277 / 255.0,
                               62.1277 / 255.0,
                               62.1277 / 255.0])
            ]),
            'Test': transforms.Compose([
                ToGrayScale(),
                Resize(dim),
                Normalize(mean=[110.0057 / 255.0,
                                110.0057 / 255.0,
                                110.0057 / 255.0],
                          std=[62.1277 / 255.0,
                               62.1277 / 255.0,
                               62.1277 / 255.0])
            ]),
        }

    train_dataset = SegDataset(data_dir + "/train/", imageFolder=imageFolder,
                                maskFolder=maskFolder, seed=100, fraction=None,
                                subset='Train', transform=data_transforms['Train'],
                                data_aug_flag=data_aug_flag)  # , dim=dim

    valid_dataset = SegDataset(data_dir + "/val/", imageFolder=imageFolder,
                               maskFolder=maskFolder, seed=100, fraction=None,
                               subset='Test', transform=data_transforms['Test'])  # , dim=dim

    test_dataset = SegDataset(data_dir + "/test/", imageFolder=imageFolder,
                               maskFolder=maskFolder, seed=100, fraction=None,
                               subset='Test', transform=data_transforms['Test'])  # , dim=dim

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, validloader, testloader