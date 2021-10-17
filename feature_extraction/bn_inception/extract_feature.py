import argparse
import os
import pickle as pkl
import time
from glob import glob

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from tsn.models import TSN


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class TSNDataSet(Dataset):
    def __init__(self, root_path, list_file, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        with open(list_file) as f:
            self.video_list = [line.strip() for line in f]
        f.close()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.video_list[index])
        images = list()
        if self.modality == 'RGB':
            num_frames = len(glob(os.path.join(video_path, "img*.jpg")))
        elif self.modality == 'Flow':
            num_frames = len(glob(os.path.join(video_path, "flow_x*.jpg")))
        for ind in (np.arange(num_frames)+1):
            images.extend(self._load_image(video_path, ind))
        process_data = self.transform(images)
        return process_data, video_path

    def __len__(self):
        return len(self.video_list)

def eval_video(data):
    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 2
    else:
        raise ValueError("Unknown modality "+args.modality)
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    baseout = np.squeeze(net(input_var).data.cpu().numpy().copy())
    return baseout


if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m', '--modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('-t', '--test_list', type=str)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--flow_prefix', type=str, default='')

    args = parser.parse_args()

    net = TSN(args.modality,
            consensus_type=args.crop_fusion_type,
            dropout=args.dropout)

    cropping = torchvision.transforms.Compose([
        GroupScale((net.input_size, net.input_size)),
    ])

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.input_dir, args.test_list,
                    modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality == 'RGB' else args.flow_prefix+"flow_{}_{:05d}.jpg",
                    transform=torchvision.transforms.Compose([
                        cropping, Stack(roll=True),
                        ToTorchFormatTensor(div=False),
                        GroupNormalize(net.input_mean, net.input_std),
                    ])),
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    for i, (data, video_path) in enumerate(data_loader):
        os.makedirs(args.output_dir, exist_ok=True)
        ft_path = os.path.join(args.output_dir, video_path[0].split(os.sep)[-1]+".pkl")
        if args.modality == 'RGB':
            length = 3
        elif args.modality == 'Flow':
            length = 2
        input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                            volatile=True)
        rst = np.squeeze(net(input_var).data.cpu().numpy().copy())
        pkl.dump(rst, open(ft_path, "wb"))
