import argparse
import os
import os.path as P
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import torchvision


class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)

class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val

class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val

class Multiply(object):
    def __init__(self, val, inverse=False) -> None:
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val

class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return np.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)

class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]

class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)


TRANSFORMS = torchvision.transforms.Compose([
    MelSpectrogram(sr=22050, nfft=1024, fmin=125, fmax=7600, nmels=80, hoplen=1024//4, spec_power=1),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
    TrimSpec(860)
])


def inv_transforms(x, folder_name='melspec_10s_22050hz'):
    '''relies on the GLOBAL contant TRANSFORMS which should be defined in this document'''
    if folder_name == 'melspec_10s_22050hz':
        i_transforms = deepcopy(TRANSFORMS.transforms[::-1])
    else:
        raise NotImplementedError
    for t in i_transforms:
        t.inverse = True
    i_transforms = torchvision.transforms.Compose(i_transforms)
    return i_transforms(x)


def get_spectrogram(audio_path, save_dir, length, folder_name='melspec_10s_22050hz', save_results=True):
    wav, _ = librosa.load(audio_path, sr=None)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]

    if folder_name == 'melspec_10s_22050hz':
        print('using', folder_name)
        mel_spec = TRANSFORMS(y)
    else:
        raise NotImplementedError

    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        audio_name = os.path.basename(audio_path).split('.')[0]
        np.save(P.join(save_dir, audio_name + '_mel.npy'), mel_spec)
        np.save(P.join(save_dir, audio_name + '_audio.npy'), y)
    else:
        return y, mel_spec



if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/features/dog/audio_10s_22050hz")
    paser.add_argument("-o", "--output_dir", default="data/features/dog/melspec_10s_22050hz")
    paser.add_argument("-l", "--length", default=220500)
    paser.add_argument("-n", '--num_worker', type=int, default=32)
    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    length = args.length

    audio_paths = glob(P.join(input_dir, "*.wav"))
    audio_paths.sort()

    with Pool(args.num_worker) as p:
        p.map(partial(
            get_spectrogram, save_dir=output_dir, length=length, folder_name=Path(output_dir).name
        ), audio_paths)
