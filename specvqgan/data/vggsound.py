import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.losses.vggishish.dataset import VGGSound as VGGSoundSpectrogramDataset
from specvqgan.modules.losses.vggishish.transforms import Crop
from train import instantiate_from_config


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

# class SelectFrames(object):
#     def __init__(self, split, feat_sample_size):
#         self.split = split
#         self.feat_sample_size = feat_sample_size
#         self.stochastic_on_train = False

#     def __call__(self, item):
#         feat_len = item['feature'].shape[0]
#         if self.stochastic_on_train and self.split == 'train':
#             idx = np.random.choice(feat_len, self.feat_sample_size)
#         else:
#             assert feat_len >= self.feat_sample_size
#             # evenly spaced points
#             idx = np.linspace(0, feat_len, self.feat_sample_size, dtype=np.int, endpoint=False)
#             # xoooxooo -> ooxooxoo
#             shift = feat_len // (self.feat_sample_size + 1)
#             idx = idx + shift
#         item['feature'] = item['feature'][idx, :]
#         return item

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


class VGGSoundSpecs(VGGSoundSpectrogramDataset):

    def __init__(self, split, spec_dir_path, mel_num=None, spec_len=None, spec_crop_len=None,
                 random_crop=None, crop_coord=None, for_which_class=None):
        super().__init__(split, spec_dir_path)
        if for_which_class:
            raise NotImplementedError
        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __getitem__(self, idx):
        # accessing VGGSoundSpectrogramDataset
        item = super().__getitem__(idx)
        # specvqgan expects `image` and `file_path_` keys in the item
        # it also expects inputs in [-1, 1] but vggsound are in [0, 1]
        item['image'] = 2 * item['input'] - 1
        item['file_path_'] = item['input_path']
        item.pop('input')
        item.pop('input_path')
        return item


class VGGSoundSpecsTrain(VGGSoundSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class VGGSoundSpecsValidation(VGGSoundSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class VGGSoundSpecsTest(VGGSoundSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class VGGSoundFeats(torch.utils.data.Dataset):

    def __init__(self, split, rgb_feats_dir_path, flow_feats_dir_path, feat_len, feat_depth, feat_crop_len,
                 replace_feats_with_random, random_crop, split_path, meta_path='./data/vggsound.csv',
                 for_which_class=None, feat_sampler_cfg=None):
        super().__init__()
        self.split = split
        self.meta_path = meta_path
        self.rgb_feats_dir_path = rgb_feats_dir_path
        self.flow_feats_dir_path = flow_feats_dir_path
        self.feat_len = feat_len
        self.feat_depth = feat_depth
        self.feat_crop_len = feat_crop_len
        self.split_path = split_path
        self.feat_sampler_cfg = feat_sampler_cfg
        self.replace_feats_with_random = replace_feats_with_random

        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        if not os.path.exists(split_path):
            raise NotImplementedError(f'The splits with clips shoud be available @ {split_path}')

        if for_which_class:
            raise NotImplementedError
        self.dataset = open(split_path).read().splitlines()

        self.feats_transforms = CropFeats([feat_crop_len, feat_depth], random_crop)

        # ResampleFrames
        self.feat_sampler = None if feat_sampler_cfg is None else instantiate_from_config(feat_sampler_cfg)

    def __getitem__(self, idx):
        item = dict()
        video_clip_name = self.dataset[idx]
        # '/path/zyTX_1BXKDE_16000_26000_mel.npy' -> 'zyTX_1BXKDE_16000_26000'
        # video_clip_name = Path(item['file_path_']).stem.replace('_mel.npy', '')
        video_name = video_clip_name[:11]

        rgb_path = os.path.join(self.rgb_feats_dir_path, f'{video_clip_name}.pkl')
        if self.replace_feats_with_random:
            rgb_feats = np.random.rand(self.feat_len, self.feat_depth//2).astype(np.float32)
        else:
            rgb_feats = pickle.load(open(rgb_path, 'rb'), encoding='bytes')
        feats = rgb_feats
        item['file_path_'] = (rgb_path, )

        # also preprocess flow
        if self.flow_feats_dir_path is not None:
            flow_path = os.path.join(self.flow_feats_dir_path, f'{video_clip_name}.pkl')
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

        target = self.video2target[video_name]
        item['target'] = target
        item['label'] = self.target2label[target]

        if self.feats_transforms is not None:
            item = self.feats_transforms(item)

        if self.feat_sampler is not None:
            item = self.feat_sampler(item)

        return item

    def __len__(self):
        return len(self.dataset)

class VGGSoundFeatsTrain(VGGSoundFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__('train', **condition_dataset_cfg)

class VGGSoundFeatsValidation(VGGSoundFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__('valid', **condition_dataset_cfg)

class VGGSoundFeatsTest(VGGSoundFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__('test', **condition_dataset_cfg)


class VGGSoundSpecsCondOnFeats(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        self.condition_dataset_cfg = condition_dataset_cfg

        self.specs_dataset = VGGSoundSpecs(split, **specs_dataset_cfg)
        self.feats_dataset = VGGSoundFeats(split, **condition_dataset_cfg)
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


class VGGSoundSpecsCondOnFeatsTrain(VGGSoundSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnFeatsValidation(VGGSoundSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnFeatsTest(VGGSoundSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('test', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnCoords(torch.utils.data.Dataset):

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

        self.specs_dataset = VGGSoundSpecs(split, **specs_dataset_cfg)

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


class VGGSoundSpecsCondOnCoordsTrain(VGGSoundSpecsCondOnCoords):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnCoordsValidation(VGGSoundSpecsCondOnCoords):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnCoordsTest(VGGSoundSpecsCondOnCoords):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('test', specs_dataset_cfg, condition_dataset_cfg)


class VGGSoundSpecsCondOnClass(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        # condition_dataset_cfg is not used anywhere else. Kept for compatibility
        self.condition_dataset_cfg = condition_dataset_cfg
        self.specs_dataset = VGGSoundSpecs(split, **specs_dataset_cfg)

    def __getitem__(self, idx):
        specs_item = self.specs_dataset[idx]
        return specs_item

    def __len__(self):
        return len(self.specs_dataset)

class VGGSoundSpecsCondOnClassTrain(VGGSoundSpecsCondOnClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnClassValidation(VGGSoundSpecsCondOnClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnClassTest(VGGSoundSpecsCondOnClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('test', specs_dataset_cfg, condition_dataset_cfg)


class VGGSoundSpecsCondOnFeatsAndClass(VGGSoundSpecsCondOnFeats):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__(split, specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnFeatsAndClassTrain(VGGSoundSpecsCondOnFeatsAndClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('train', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnFeatsAndClassValidation(VGGSoundSpecsCondOnFeatsAndClass):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__('valid', specs_dataset_cfg, condition_dataset_cfg)

class VGGSoundSpecsCondOnFeatsAndClassTest(VGGSoundSpecsCondOnFeatsAndClass):
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
    import sys

    from omegaconf import OmegaConf

    # SPECTROGRAMS + FEATS (Subsample)
    cfg = OmegaConf.load('./configs/vggsound_transformer.yaml')
    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets['train']))
    print(data.datasets['train'][24])
    print(data.datasets['validation'][24])
    print(data.datasets['test'][24])
    print(data.datasets['train'][24]['feature'].shape)
