import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.losses.vggishish.transforms import Crop
from train import instantiate_from_config


def make_split_files(split_path, feat_folder, feat_suffix):
    # feat_suffix is e.g. '_mel.npy' or '.pkl
    # output: train and valid files with rows like `<class>/video_<id>`
    train_dataset = []
    valid_dataset = []
    filepaths = sorted(glob(os.path.join(feat_folder, '*' + feat_suffix)))
    assert len(filepaths) > 0, 'Empty filelist'
    # [`<class>/video_<id>`]
    classes = [Path(f).parent.parent.stem for f in filepaths]
    vid_ids = [Path(f).name.replace(feat_suffix, '') for f in filepaths]

    for cls in sorted(list(set(classes))):
        n_valid = 128 if cls in ['dog', 'fireworks', 'baby', 'drum'] else 64
        cls_dataset = []
        for c, v in zip(classes, vid_ids):
            if c == cls:
                cls_dataset.append(f'{c}/{v}')
        train_dataset.extend(cls_dataset[:-n_valid])
        valid_dataset.extend(cls_dataset[-n_valid:])

    save_train_path = Path(split_path).parent / 'vas_train.txt'
    save_valid_path = Path(split_path).parent / 'vas_valid.txt'

    with open(save_train_path, 'w') as outf:
        for row in train_dataset:
            outf.write(f'{row}\n')
    with open(save_valid_path, 'w') as outf:
        for row in valid_dataset:
            outf.write(f'{row}\n')

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class CropFeats(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['feature'] = self.preprocessor(image=item['feature'])['image']
        return item

class CropCoords(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['coord'] = self.preprocessor(image=item['coord'])['image']
        return item


class ResampleFrames(object):
    def __init__(self, feat_sample_size, times_to_repeat_after_resample=None):
        self.feat_sample_size = feat_sample_size
        self.times_to_repeat_after_resample = times_to_repeat_after_resample

    def __call__(self, item):
        feat_len = item['feature'].shape[0]

        ## resample
        assert feat_len >= self.feat_sample_size
        # evenly spaced points (abcdefghkl -> aoooofoooo)
        idx = np.linspace(0, feat_len, self.feat_sample_size, dtype=np.int, endpoint=False)
        # xoooo xoooo -> ooxoo ooxoo
        shift = feat_len // (self.feat_sample_size + 1)
        idx = idx + shift

        ## repeat after resampling (abc -> aaaabbbbcccc)
        if self.times_to_repeat_after_resample is not None and self.times_to_repeat_after_resample > 1:
            idx = np.repeat(idx, self.times_to_repeat_after_resample)

        item['feature'] = item['feature'][idx, :]
        return item

class VASSpecs(torch.utils.data.Dataset):

    def __init__(self, split, spec_dir_path, mel_num=None, spec_len=None, spec_crop_len=None,
                 random_crop=None, crop_coord=None, for_which_class=None):
        super().__init__()
        self.split = split
        self.spec_dir_path = spec_dir_path
        # fixing split_path in here because of compatibility with vggsound which hangles it in vggishish
        self.split_path = f'./data/vas_{split}.txt'
        self.feat_suffix = '_mel.npy'

        if not os.path.exists(self.split_path):
            print(f'split does not exist in {self.split_path}. Creating new ones...')
            make_split_files(self.split_path, spec_dir_path, self.feat_suffix)

        full_dataset = open(self.split_path).read().splitlines()
        # ['baby/video_00000', ..., 'dog/video_00000', ...]
        if for_which_class:
            self.dataset = [v for v in full_dataset if v.startswith(for_which_class)]
        else:
            self.dataset = full_dataset

        unique_classes = sorted(list(set([cls_vid.split('/')[0] for cls_vid in self.dataset])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}

        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __getitem__(self, idx):
        item = {}

        cls, vid = self.dataset[idx].split('/')
        spec_path = os.path.join(self.spec_dir_path.replace('*', cls), f'{vid}{self.feat_suffix}')

        spec = np.load(spec_path)
        item['input'] = spec
        item['file_path_'] = spec_path

        item['label'] = cls
        item['target'] = self.label2target[cls]

        if self.transforms is not None:
            item = self.transforms(item)

        # specvqgan expects `image` and `file_path_` keys in the item
        # it also expects inputs in [-1, 1] but specs are in [0, 1]
        item['image'] = 2 * item['input'] - 1
        item.pop('input')

        return item

    def __len__(self):
        return len(self.dataset)


class VASSpecsTrain(VASSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class VASSpecsValidation(VASSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class VASSpecsTest(VASSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class VASFeats(torch.utils.data.Dataset):

    def __init__(self, split, rgb_feats_dir_path, flow_feats_dir_path, feat_len, feat_depth, feat_crop_len,
                 replace_feats_with_random, random_crop, split_path, for_which_class, feat_sampler_cfg):
        super().__init__()
        self.split = split
        self.rgb_feats_dir_path = rgb_feats_dir_path
        self.flow_feats_dir_path = flow_feats_dir_path
        self.feat_len = feat_len
        self.feat_depth = feat_depth
        self.feat_crop_len = feat_crop_len
        self.split_path = split_path
        self.feat_suffix = '.pkl'
        self.feat_sampler_cfg = feat_sampler_cfg
        self.replace_feats_with_random = replace_feats_with_random

        if not os.path.exists(split_path):
            print(f'split does not exist in {split_path}. Creating new ones...')
            make_split_files(split_path, rgb_feats_dir_path, self.feat_suffix)

        full_dataset = open(split_path).read().splitlines()
        if for_which_class:
            # ['baby/video_00000', ..., 'dog/video_00000', ...]
            self.dataset = [v for v in full_dataset if v.startswith(for_which_class)]
        else:
            self.dataset = full_dataset

        unique_classes = sorted(list(set([cls_vid.split('/')[0] for cls_vid in self.dataset])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}

        self.feats_transforms = CropFeats([feat_crop_len, feat_depth], random_crop)
        # self.normalizer = StandardNormalizeFeats(rgb_feats_dir_path, flow_feats_dir_path, feat_len)
        # ResampleFrames
        self.feat_sampler = None if feat_sampler_cfg is None else instantiate_from_config(feat_sampler_cfg)

    def __getitem__(self, idx):
        item = dict()
        cls, vid = self.dataset[idx].split('/')

        rgb_path = os.path.join(self.rgb_feats_dir_path.replace('*', cls), f'{vid}{self.feat_suffix}')
        # just a dummy random features acting like a fake interface for no features experiment
        if self.replace_feats_with_random:
            rgb_feats = np.random.rand(self.feat_len, self.feat_depth//2).astype(np.float32)
        else:
            rgb_feats = pickle.load(open(rgb_path, 'rb'), encoding='bytes')
        feats = rgb_feats
        item['file_path_'] = (rgb_path, )

        # also preprocess flow
        if self.flow_feats_dir_path is not None:
            flow_path = os.path.join(self.flow_feats_dir_path.replace('*', cls), f'{vid}{self.feat_suffix}')
            # just a dummy random features acting like a fake interface for no features experiment
            if self.replace_feats_with_random:
                flow_feats = np.random.rand(self.feat_len, self.feat_depth//2).astype(np.float32)
            else:
                flow_feats = pickle.load(open(flow_path, 'rb'), encoding='bytes')
            # (T, 2*D)
            feats = np.concatenate((rgb_feats, flow_feats), axis=1)
            item['file_path_'] = (rgb_path, flow_path)

        # pad or trim
        feats_padded = np.zeros((self.feat_len, feats.shape[1]))
        feats_padded[:feats.shape[0], :] = feats[:self.feat_len, :]
        item['feature'] = feats_padded

        item['label'] = cls
        item['target'] = self.label2target[cls]

        if self.feats_transforms is not None:
            item = self.feats_transforms(item)

        if self.feat_sampler is not None:
            item = self.feat_sampler(item)

        return item

    def __len__(self):
        return len(self.dataset)


# class VGGSoundFeatsTrain(VASFeats):
#     def __init__(self, condition_dataset_cfg):
#         super().__init__('train', **condition_dataset_cfg)

# class VGGSoundFeatsValidation(VASFeats):
#     def __init__(self, condition_dataset_cfg):
#         super().__init__('valid', **condition_dataset_cfg)

# class VGGSoundFeatsTest(VASFeats):
#     def __init__(self, condition_dataset_cfg):
#         super().__init__('test', **condition_dataset_cfg)


class VASSpecsCondOnFeats(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        self.condition_dataset_cfg = condition_dataset_cfg

        self.specs_dataset = VASSpecs(split, **specs_dataset_cfg)
        self.feats_dataset = VASFeats(split, **condition_dataset_cfg)
        assert len(self.specs_dataset) == len(self.feats_dataset)

    def __getitem__(self, idx):
        specs_item = self.specs_dataset[idx]
        feats_item = self.feats_dataset[idx]

        # sanity check and removing those from one of the dicts
        for key in ['target', 'label']:
            assert specs_item[key] == feats_item[key]
            feats_item.pop(key)

        # keeping both sets of paths to features
        specs_item['file_path_specs_'] = specs_item.pop('file_path_')
        feats_item['file_path_feats_'] = feats_item.pop('file_path_')

        # merging both dicts
        specs_feats_item = dict(**specs_item, **feats_item)

        return specs_feats_item

    def __len__(self):
        return len(self.specs_dataset)


class VASSpecsCondOnFeatsTrain(VASSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnFeatsValidation(VASSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)


class VASSpecsCondOnCoords(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        self.condition_dataset_cfg = condition_dataset_cfg

        self.crop_coord = self.specs_dataset_cfg.crop_coord
        if self.crop_coord:
            print('DID YOU EXPECT THAT COORDS ARE CROPPED NOW?')
            self.F = self.specs_dataset_cfg.mel_num
            self.T = self.specs_dataset_cfg.spec_len
            self.T_crop = self.specs_dataset_cfg.spec_crop_len
            self.transforms = CropCoords([self.F, self.T_crop], self.specs_dataset_cfg.random_crop)

        self.specs_dataset = VASSpecs(split, **specs_dataset_cfg)

    def __getitem__(self, idx):
        specs_item = self.specs_dataset[idx]
        if self.crop_coord:
            coord = np.arange(self.F * self.T).reshape(self.T, self.F) / (self.T * self.F)
            coord = coord.T
            specs_item['coord'] = coord
            specs_item = self.transforms(specs_item)
        else:
            F, T = specs_item['image'].shape
            coord = np.arange(F * T).reshape(T, F) / (T * F)
            coord = coord.T
            specs_item['coord'] = coord

        return specs_item

    def __len__(self):
        return len(self.specs_dataset)


class VASSpecsCondOnCoordsTrain(VASSpecsCondOnCoords):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnCoordsValidation(VASSpecsCondOnCoords):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)


class VASSpecsCondOnClass(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        # not used anywhere else. Kept for compatibility
        self.condition_dataset_cfg = condition_dataset_cfg
        self.specs_dataset = VASSpecs(split, **specs_dataset_cfg)

    def __getitem__(self, idx):
        specs_item = self.specs_dataset[idx]
        return specs_item

    def __len__(self):
        return len(self.specs_dataset)

class VASSpecsCondOnClassTrain(VASSpecsCondOnClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnClassValidation(VASSpecsCondOnClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)


class VASSpecsCondOnFeatsAndClass(VASSpecsCondOnFeats):
    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__(split, specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnFeatsAndClassTrain(VASSpecsCondOnFeatsAndClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnFeatsAndClassValidation(VASSpecsCondOnFeatsAndClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)

class VASSpecsCondOnFeatsAndClassTest(VASSpecsCondOnFeatsAndClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('test', specs_dataset_cfg, condition_dataset_cfg)

# class StandardNormalizeFeats(object):
#     def __init__(self, rgb_feats_dir_path, flow_feats_dir_path, feat_len,
#                  train_ids_path='./data/vggsound_test.txt', cache_path='./data/'):
#         self.rgb_feats_dir_path = rgb_feats_dir_path
#         self.flow_feats_dir_path = flow_feats_dir_path
#         self.train_ids_path = train_ids_path
#         self.feat_len = feat_len
#         # making the stats filename to match the specs dir name
#         self.cache_path = os.path.join(
#             cache_path, f'train_means_stds_{Path(rgb_feats_dir_path).stem}.txt'.replace('_rgb', '')
#         )
#         logger.info('Assuming that the input stats are calculated using preprocessed spectrograms (log)')
#         self.train_stats = self.calculate_or_load_stats()

#     def __call__(self, rgb_flow_feats):
#         return (rgb_flow_feats - self.train_stats['means']) / self.train_stats['stds']

#     def calculate_or_load_stats(self):
#         try:
#             # (F, 2)
#             train_stats = np.loadtxt(self.cache_path)
#             means, stds = train_stats.T
#             logger.info('Trying to load train stats for Standard Normalization of inputs')
#         except OSError:
#             logger.info('Could not find the precalculated stats for Standard Normalization. Calculating...')
#             train_vid_ids = open(self.train_ids_path).read().splitlines()
#             means = [None] * len(train_vid_ids)
#             stds = [None] * len(train_vid_ids)
#             for i, vid_id in enumerate(tqdm(train_vid_ids)):
#                 rgb_path = os.path.join(self.rgb_feats_dir_path, f'{vid_id}.pkl')
#                 flow_path = os.path.join(self.flow_feats_dir_path, f'{vid_id}.pkl')
#                 with open(rgb_path, 'rb') as f:
#                     rgb_feats = pickle.load(f, encoding='bytes')
#                 with open(flow_path, 'rb') as f:
#                     flow_feats = pickle.load(f, encoding='bytes')
#                 # (T, 2*D)
#                 feats = np.concatenate((rgb_feats, flow_feats), axis=1)
#                 # pad or trim
#                 feats_padded = np.zeros((self.feat_len, feats.shape[1]))
#                 feats_padded[:feats.shape[0], :] = feats[:self.feat_len, :]
#                 feats = feats_padded
#                 means[i] = feats.mean(axis=1)
#                 stds[i] = feats.std(axis=1)
#             # (F) <- (num_files, D)
#             means = np.array(means).mean(axis=0)
#             stds = np.array(stds).mean(axis=0)
#             # saving in two columns
#             np.savetxt(self.cache_path, np.vstack([means, stds]).T, fmt='%0.8f')
#         means = means.reshape(-1, 1)
#         stds = stds.reshape(-1, 1)
#         assert 'train' in self.train_ids_path
#         return {'means': means, 'stds': stds}


if __name__ == '__main__':
    from omegaconf import OmegaConf

    # SPECTROGRAMS + FEATURES
    cfg = OmegaConf.load('./configs/vas_transformer.yaml')
    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()
    print(data.datasets['train'][24])
    print(data.datasets['validation'][24])
    print(data.datasets['validation'][-1]['feature'].shape)
    print(data.datasets['validation'][-1]['image'].shape)
