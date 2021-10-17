import scipy.io.wavfile
from mel2wav.extract_mel_spectrogram import TRANSFORMS
import torch


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)

def wav2mel(batch, wave_len=None):

    if len(batch.shape) == 3:
        assert batch.shape[1] == 1, 'Multi-channel audio?'
        batch = batch.squeeze(1)

    batch = torch.stack([torch.from_numpy(TRANSFORMS(e.numpy())) for e in batch.cpu()]).float()

    if wave_len is not None:
        batch = batch[:, :, :wave_len]

    return batch
