'''
The code is partially borrowed from:
https://github.com/v-iashin/video_features/blob/861efaa4ed67/utils/utils.py
and
https://github.com/PeihaoChen/regnet/blob/199609/extract_audio_and_video.py
'''
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from omegaconf.omegaconf import OmegaConf
from sample_visualization import (load_feature_extractor,
                                  load_model_from_config, load_vocoder)
from specvqgan.data.vggsound import CropFeats
from specvqgan.util import download, md5_hash
from specvqgan.models.cond_transformer import disabled_train
from train import instantiate_from_config

from feature_extraction.extract_mel_spectrogram import get_spectrogram

plt.rcParams['savefig.bbox'] = 'tight'


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path


def check_video_for_audio(path):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -loglevel error -show_entries stream=codec_type -of default=nw=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    print(result)
    return 'codec_type=audio' in result

def get_duration(path):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -hide_banner -loglevel panic' \
          f' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout.decode('utf-8').replace('\n', ''))
    return duration

def trim_video(video_path: str, start: int, trim_duration: int = 10, tmp_path: str = './tmp'):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    if Path(video_path).suffix != '.mp4':
        print(f'File Extension is not `mp4` (it is {Path(video_path).suffix}). It will be re-encoded to mp4.')

    video_duration = get_duration(video_path)
    print('Video Duration:', video_duration)
    assert video_duration > start, f'Video Duration < Trim Start: {video_duration} < {start}'

    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)
    trim_vid_path = os.path.join(tmp_path, f'{Path(video_path).stem}_trim_to_{trim_duration}s.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic' \
          f' -i {video_path} -ss {start} -t {trim_duration} -y {trim_vid_path}'
    subprocess.call(cmd.split())
    print('Trimmed the input video', video_path, 'and saved the output @', trim_vid_path)

    return trim_vid_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -i {video_path} -filter:v fps=fps={extraction_fps} {new_path}'
    subprocess.call(cmd.split())

    return new_path

def maybe_download_model(model_name: str, log_dir: str) -> str:
    name2info = {
        '2021-06-20T16-35-20_vggsound_transformer': {
            'info': 'No Feats',
            'hash': 'b1f9bb63d831611479249031a1203371',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-20T16-35-20_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-03-22_vggsound_transformer': {
            'info': '1 ResNet50 Feature',
            'hash': '27a61d4b74a72578d13579333ed056f6',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-03-22_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-25_vggsound_transformer': {
            'info': '5 ResNet50 Features',
            'hash': 'f4d7105811589d441b69f00d7d0b8dc8',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-25_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-41_vggsound_transformer': {
            'info': '212 ResNet50 Features',
            'hash': 'b222cc0e7aeb419f533d5806a08669fe',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-41_vggsound_transformer.tar.gz',
        },
        '2021-06-03T00-43-28_vggsound_transformer': {
            'info': 'Class Label',
            'hash': '98a3788ab973f1c3cc02e2e41ad253bc',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-03T00-43-28_vggsound_transformer.tar.gz',
        },
        '2021-05-19T22-16-54_vggsound_codebook': {
            'info': 'VGGSound Codebook',
            'hash': '7ea229427297b5d220fb1c80db32dbc5',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-05-19T22-16-54_vggsound_codebook.tar.gz',
        }
    }
    print(f'Using: {model_name} ({name2info[model_name]["info"]})')
    model_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(model_dir):
        tar_local_path = os.path.join(log_dir, f'{model_name}.tar.gz')
        # check if tar already exists and its md5sum
        if not os.path.exists(tar_local_path) or md5_hash(tar_local_path) != name2info[model_name]['hash']:
            down_link = name2info[model_name]['link']
            download(down_link, tar_local_path)
            print('Unpacking', tar_local_path, 'to', log_dir)
            shutil.unpack_archive(tar_local_path, log_dir)
            # clean-up space as we already have unpacked folder
            os.remove(tar_local_path)
    return model_dir

def load_config(model_dir: str):
    # Load the config
    config_main = sorted(glob(os.path.join(model_dir, 'configs/*-project.yaml')))[-1]
    config_pylt = sorted(glob(os.path.join(model_dir, 'configs/*-lightning.yaml')))[-1]
    config = OmegaConf.merge(
        OmegaConf.load(config_main),
        OmegaConf.load(config_pylt),
    )
    # patch config. E.g. if the model is trained on another machine with different paths
    for a in ['spec_dir_path', 'rgb_feats_dir_path', 'flow_feats_dir_path']:
        if config.data.params[a] is not None:
            if 'vggsound.VGGSound' in config.data.params.train.target:
                base_path = './data/vggsound/'
            elif 'vas.VAS' in config.data.params.train.target:
                base_path = './data/vas/features/*/'
            else:
                raise NotImplementedError
            config.data.params[a] = os.path.join(base_path, Path(config.data.params[a]).name)
    return config

def load_model(model_name, log_dir, device):
    to_use_gpu = True if device.type == 'cuda' else False
    model_dir = maybe_download_model(model_name, log_dir)
    config = load_config(model_dir)

    # Sampling model
    ckpt = sorted(glob(os.path.join(model_dir, 'checkpoints/*.ckpt')))[-1]
    pl_sd = torch.load(ckpt, map_location='cpu')
    sampler = load_model_from_config(config.model, pl_sd['state_dict'], to_use_gpu)['model']
    sampler.to(device)

    # aux models (vocoder and melception)
    ckpt_melgan = config.lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder
    melgan = load_vocoder(ckpt_melgan, eval_mode=True)['model'].to(device)
    melception = load_feature_extractor(to_use_gpu, eval_mode=True)
    return config, sampler, melgan, melception

def load_model_vocoder(device):
    to_use_gpu = True if device.type == 'cuda' else False
#     model_dir = maybe_download_model(model_name, log_dir)
#     config = load_config(model_dir)
    
    # vocoder only model (vocoder)
    #ckpt_melgan = config.lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder
    ckpt_melgan = '/content/SpecVQGAN/vocoder/logs/vggsound/best_netG.pt'
    melgan = load_vocoder(ckpt_melgan, eval_mode=True)['model'].to(device)
    return melgan

def load_neural_audio_codec(model_name, log_dir, device):
    model_dir = maybe_download_model(model_name, log_dir)
    config = load_config(model_dir)

    config.model.params.ckpt_path = f'./logs/{model_name}/checkpoints/last.ckpt'
    print(config.model.params.ckpt_path)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model = model.eval()
    model.train = disabled_train
    vocoder = load_vocoder(Path('./vocoder/logs/vggsound/'), eval_mode=True)['model'].to(device)
    return config, model, vocoder

class LeftmostCropOrTile(object):
    def __init__(self, crop_or_tile_to):
        self.crop_or_tile_to = crop_or_tile_to

    def __call__(self, item: Dict):
        # tile or crop features to the `crop_or_tile_to`
        T, D = item['feature'].shape
        if T != self.crop_or_tile_to:
            how_many_tiles_needed = 1 + (self.crop_or_tile_to // T)
            item['feature'] = np.tile(item['feature'], (how_many_tiles_needed, 1))[:self.crop_or_tile_to, :]
        return item

class ExtractResNet50(torch.nn.Module):

    def __init__(self, extraction_fps, feat_cfg, device, batch_size=32, tmp_dir='./tmp'):
        super(ExtractResNet50, self).__init__()
        self.tmp_path = tmp_dir
        self.extraction_fps = extraction_fps
        self.batch_size = batch_size
        self.feat_cfg = feat_cfg

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.stds)
        ])
        random_crop = False
        self.post_transforms = transforms.Compose([
            LeftmostCropOrTile(feat_cfg.feat_len),
            CropFeats([feat_cfg.feat_crop_len, feat_cfg.feat_depth], random_crop),
            (lambda x: x) if feat_cfg.feat_sampler_cfg is None else instantiate_from_config(feat_cfg.feat_sampler_cfg),
        ])
        self.device = device
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        self.model_class = self.model.fc
        self.model.fc = torch.nn.Identity()

    @torch.no_grad()
    def forward(self, video_path: str) -> Dict[str, np.ndarray]:

        if self.feat_cfg.replace_feats_with_random:
            T, D = self.feat_cfg.feat_sampler_cfg.params.feat_sample_size, self.feat_cfg.feat_depth
            print(f'Since we are in "No Feats" setting, returning a random feature: [{T}, {D}]')
            random_features = {'feature': torch.rand(T, D)}
            return random_features, []

        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        cap = cv2.VideoCapture(video_path)
        batch_list = []
        vid_feats = []
        cached_frames = []
        transforms_for_show = transforms.Compose(self.transforms.transforms[:4])
        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True

        # iterating through the opened video frame-by-frame and occationally run the model once a batch is
        # formed
        while cap.isOpened():
            frame_exists, rgb = cap.read()

            if first_frame and not frame_exists:
                continue
            first_frame = False

            if frame_exists:
                # prepare data and cache if needed
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cached_frames.append(transforms_for_show(rgb))
                rgb = self.transforms(rgb).unsqueeze(0).to(self.device)
                batch_list.append(rgb)
                # when batch is formed to inference
                if len(batch_list) == self.batch_size:
                    batch_feats = self.model(torch.cat(batch_list))
                    vid_feats.extend(batch_feats.tolist())
                    # clean up the batch list
                    batch_list = []
            else:
                # if the last batch was smaller than the batch size, we still need to process those frames
                if len(batch_list) != 0:
                    batch_feats = self.model(torch.cat(batch_list))
                    vid_feats.extend(batch_feats.tolist())
                cap.release()
                break

        vid_feats = np.array(vid_feats)
        features = {'feature': vid_feats}
        print('Raw Extracted Representation:', features['feature'].shape)

        if self.post_transforms is not None:
            features = self.post_transforms(features)
            # using 'feature' as the key to reuse the feature resampling transform
            cached_frames = self.post_transforms.transforms[-1]({'feature': torch.stack(cached_frames)})['feature']

        print('Post-processed Representation:', features['feature'].shape)

        return features, cached_frames

def extract_melspectrogram(in_path: str, sr: int, duration: int = 10, tmp_path: str = './tmp') -> np.ndarray:
    '''Extract Melspectrogram similar to RegNet.'''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert in_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # Extract audio from a video if needed
    if in_path.endswith('.wav'):
        audio_raw = in_path
    else:
        audio_raw = os.path.join(tmp_path, f'{Path(in_path).stem}.wav')
        cmd = f'{which_ffmpeg()} -i {in_path} -hide_banner -loglevel panic -f wav -vn -y {audio_raw}'
        subprocess.call(cmd.split())

    # Extract audio from a video
    audio_new = os.path.join(tmp_path, f'{Path(in_path).stem}_{sr}hz.wav')
    cmd = f'{which_ffmpeg()} -i {audio_raw} -hide_banner -loglevel panic -ac 1 -ab 16k -ar {sr} -y {audio_new}'
    subprocess.call(cmd.split())

    length = int(duration * sr)
    audio_zero_pad, spec = get_spectrogram(audio_new, save_dir=None, length=length, save_results=False)

    # specvqgan expects inputs to be in [-1, 1] but spectrograms are in [0, 1]
    spec = 2 * spec - 1

    return spec


def show_grid(imgs):
    print('Rendering the Plot with Frames Used in Conditioning')
    figsize = ((imgs.shape[1] // 228 + 1) * 5, (imgs.shape[2] // 228 + 1) * 5)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def calculate_codebook_bitrate(duration, quant_z, codebook_size):
    # Calculating the Bitrate
    bottle_neck_size = quant_z.shape[-2:]
    bits_per_codebook_entry = (codebook_size-1).bit_length()
    bitrate = bits_per_codebook_entry * bottle_neck_size.numel() / duration / 1024
    print(f'The input audio is {duration:.2f} seconds long.')
    print(f'Codebook size is {codebook_size} i.e. a codebook entry allocates {bits_per_codebook_entry} bits')
    print(f'SpecVQGAN bottleneck size: {list(bottle_neck_size)}')
    print(f'Thus, bitrate is {bitrate:.2f} kbps')
    return bitrate

def get_audio_file_bitrate(file):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -v error -select_streams a:0'\
          f' -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 {file}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    bitrate = int(result.stdout.decode('utf-8').replace('\n', ''))
    bitrate /= 1024
    return bitrate


if __name__ == '__main__':
    # if empty, it wasn't found
    print(which_ffmpeg())
