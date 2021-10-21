import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:
    pass

import torch
import torchvision
import yaml
from omegaconf import OmegaConf

from specvqgan.util import get_ckpt_path

sys.path.insert(0, '.')  # nopep8
import matplotlib.pyplot as plt
import soundfile
from torch.utils.data.dataloader import default_collate

from feature_extraction.extract_mel_spectrogram import inv_transforms
from train import instantiate_from_config
from vocoder.modules import Generator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        '--vocoder_path',
        default='./vocoder/logs/vggsound/',
        help='The path to the folder with pre-trained Vocoder (a folder from ./vocoder/logs)'
    )
    parser.add_argument(
        '--logdir',
        default='./logs/',
        help='Path to the log dir with pre-trained GPT'
    )
    return parser

def rename_models(x):
    x = x[x.index('T')+1:]
    name2type = {
        '00-43-28_vggsound_transformer': 'VGGSound – Class – VGGSound Codebook',
        '14-41-19_vas_transformer': 'VAS – Class – VGGSound Codebook',
        '09-42-07_vas_transformer': 'VAS – Class – VAS Codebook',
        '16-35-20_vggsound_transformer': 'VGGSound – No Feats – VGGSound Codebook',
        '11-18-51_vggsound_transformer': 'VGGSound – 1 Feat BN – VGGSound Codebook',
        '09-34-10_vggsound_transformer': 'VGGSound – 5 Feats BN – VGGSound Codebook',
        '07-27-58_vggsound_transformer': 'VGGSound – 212 Feats BN – VGGSound Codebook',
        '16-34-36_vas_transformer': 'VAS – No Feats – VGGSound Codebook',
        '06-32-51_vas_transformer': 'VAS – 1 Feat BN – VGGSound Codebook',
        '05-51-34_vas_transformer': 'VAS – 5 Feats BN – VGGSound Codebook',
        '05-38-40_vas_transformer': 'VAS – 212 Feats BN – VGGSound Codebook',
        '16-24-38_vas_transformer': 'VAS – No Feats – VAS Codebook',
        '13-31-37_vas_transformer': 'VAS – 1 Feats BN – VAS Codebook',
        '14-14-24_vas_transformer': 'VAS – 5 Feats BN – VAS Codebook',
        '15-17-18_vas_transformer': 'VAS – 212 Feats BN – VAS Codebook',
        '11-47-40_vas_transformer': 'VAS – 1 Feat RN50 – VGGSound Codebook',
        '11-36-00_vas_transformer': 'VAS – 5 Feats RN50 – VGGSound Codebook',
        '11-52-28_vas_transformer': 'VAS – 212 Feats RN50 – VGGSound Codebook',
        '14-59-49_vas_transformer': 'VAS – 1 Feat RN50 – VAS Codebook',
        '14-51-25_vas_transformer': 'VAS – 5 Feats RN50 – VAS Codebook',
        '13-34-39_vas_transformer': 'VAS – 212 Feats RN50 – VAS Codebook',
        '21-03-22_vggsound_transformer': 'VGGSound – 1 Feat RN50 – VGGSound Codebook',
        '21-34-25_vggsound_transformer': 'VGGSound – 5 Feats RN50 – VGGSound Codebook',
        '21-34-41_vggsound_transformer': 'VGGSound – 212 Feats RN50 – VGGSound Codebook',
    }
    if x in name2type:
        x = f'{name2type[x]} ({x})'
    return x

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        st.warning("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        st.warning("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            st.warning("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            st.warning("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        try:
            st.warning(f"Missing Keys in State Dict: {missing}")
            st.warning(f"Unexpected Keys in State Dict: {unexpected}")
        except NameError:
            pass
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def load_vocoder(ckpt_vocoder: str, eval_mode: bool):
    ckpt_vocoder = Path(ckpt_vocoder)
    vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')

    with open(ckpt_vocoder / 'args.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader)

    vocoder = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
    vocoder.load_state_dict(vocoder_sd)

    if eval_mode:
        vocoder.eval()

    return {'model': vocoder}

def load_feature_extractor(gpu, eval_mode=True):
    s = '''
    feature_extractor:
        target: evaluation.feature_extractors.melception.Melception
        params:
            num_classes: 309
            features_list: ['logits']
            feature_extractor_weights_path: ./evaluation/logs/21-05-10T09-28-40/melception-21-05-10T09-28-40.pt
    transform_dset_out_to_inception_in:
    - target: evaluation.datasets.transforms.FromMinusOneOneToZeroOne
    - target: specvqgan.modules.losses.vggishish.transforms.StandardNormalizeAudio
      params:
        specs_dir: ./data/vggsound/melspec_10s_22050hz
        cache_path: ./specvqgan/modules/losses/vggishish/data/
    - target: evaluation.datasets.transforms.GetInputFromBatchByKey
      params:
        input_key: image
    - target: evaluation.datasets.transforms.ToFloat32'''
    feat_extractor_cfg = OmegaConf.create(s)
    # downloading the checkpoint for melception
    get_ckpt_path('melception', 'evaluation/logs/21-05-10T09-28-40')
    pl_sd = torch.load(feat_extractor_cfg.feature_extractor.params.feature_extractor_weights_path,
                       map_location="cpu")

    # use gpu=False to compute it on CPU
    feat_extractor = load_model_from_config(
        feat_extractor_cfg.feature_extractor, pl_sd['model'], gpu=gpu, eval_mode=eval_mode)['model']

    if feat_extractor_cfg.transform_dset_out_to_inception_in is not None:
        transforms = [instantiate_from_config(c) for c in feat_extractor_cfg.transform_dset_out_to_inception_in]
    else:
        transforms = [lambda x: x]
    transforms = torchvision.transforms.Compose(transforms)

    vggsound_meta = list(csv.reader(open('./data/vggsound.csv'), quotechar='"'))
    unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
    label2target = {label: target for target, label in enumerate(unique_classes)}
    target2label = {target: label for label, target in label2target.items()}
    return {'model': feat_extractor, 'transforms': transforms, 'target2label': target2label}

def load_model_and_dataset(config, ckpt, ckpt_vocoder, gpu=True, eval_mode=True):
    # get data
    dsets = instantiate_from_config(config.data)
    dsets.prepare_data()
    dsets.setup()

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    # loading the vocoder
    if ckpt_vocoder:
        vocoder = load_vocoder(ckpt_vocoder, eval_mode)['model']
        vocoder = vocoder.to('cuda') if gpu else vocoder

    model = load_model_from_config(config.model, pl_sd['state_dict'], gpu=gpu, eval_mode=eval_mode)['model']

    # patch config for the adjusted input length which could be longer than during training (infinite samples)
    # local_permuter = model.first_stage_permuter
    # if config.model.params.first_stage_permuter_config.params.W is not None:
    #     config.model.params.first_stage_permuter_config.params.W *= W_scale
    #     model.first_stage_permuter = instantiate_from_config(config.model.params.first_stage_permuter_config).cuda().eval()
    #     print(config.model.params.first_stage_permuter_config)

    feat_extractor = load_feature_extractor(gpu, eval_mode)

    return dsets, model, vocoder, global_step, feat_extractor


# the same as the decorator `@st.cache(allow_output_mutation=True, suppress_st_warning=True)`
try:
    load_model_and_dataset = st.cache(load_model_and_dataset, allow_output_mutation=True,
                                      suppress_st_warning=True)
except NameError:
    pass


def bchw_to_st(x, to_scale=True, flip_dims=None):
    if flip_dims is not None:
        # dims is a tuple. To flip only 2nd dim use: `flip_dims=(2,)`
        x = x.flip(dims=flip_dims)
    if to_scale:
        # (-1, 1) -> (0, 1)
        return (x.detach().cpu().numpy().transpose(0, 2, 3, 1) + 1.) / 2.
    else:
        return x.detach().cpu().numpy().transpose(0, 2, 3, 1)

def tensor_to_plt(x, vmin=None, vmax=None, flip_dims=None):
    if flip_dims is not None:
        # dims is a tuple. To flip only 2nd dim use: `flip_dims=(2,)`
        x = x.flip(dims=flip_dims)
    # remove batch dim and make channel-last
    if len(x.shape) > 3:
        x = x.squeeze(0)
    # if the figure is taller than it is wider rotate (transpose). Also clipping it as feats can be large
    if x.shape[-1] < x.shape[-2]:
        x = x.clip(-2, 2).transpose(-1, -2)
    x = x.cpu()
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0)
    # fig, arr = plt.subplots(nrows=1, ncols=1)
    # # arr[i].set_title(f'{vid_name}_{name}')
    # arr.imshow(x)
    # arr.set_frame_on(False)
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    # for facehq
    # TODO: if x.shape[0] == 3:
    #     x = x.flip(dims=(1,)).permute(1, 2, 0)
    #     x = (x + 1) / 2
    #     x = x.clip(0, 1)

    # newer version of the matplotlib started to fails when an image has 3 dim with `1` as the last one
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    ax.imshow(x, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
    # ax.set_title('Some', fontsize=8)
    return fig

def save_results(spec_plt, waves_dict, topk_preds, logdir, batch, mode, sample_rate, specs_key_in_batch):
    # implemented only for B=1, otherwise mind the batch[key][0]
    label = ''.join(filter(lambda x: str.isalnum(x) or ' ', batch['label'][0])).replace(' ', '_')
    target = int(batch['target'][0])
    vid_id = Path(batch[specs_key_in_batch][0]).name.replace('_mel.npy', '')
    time_stamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir = Path(logdir) / 'streamlit' / f'{target:03d}_{label}' / vid_id
    os.makedirs(save_dir, exist_ok=True)
    dpi = 300
    for wave_type, wave in waves_dict.items():
        soundfile.write(save_dir / f'{mode}_{time_stamp}_{wave_type}.wav', wave, sample_rate, 'PCM_24')
        if len(wave) > sample_rate * 10:
            dpi *= 10
    spec_plt.savefig(save_dir / f'{mode}_{time_stamp}.png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    with open(save_dir / f'{mode}_{time_stamp}_topkpreds.txt', 'w') as out_f:
        out_f.write(topk_preds)

def show_wave_in_streamlit(wave_npy, sample_rate, caption):
    # showing in streamlit. We cannot just show the npy wave and we need to save it first
    temp_wav_file_path = 'todel.wav'
    soundfile.write(temp_wav_file_path, wave_npy, sample_rate, 'PCM_24')
    st.text(caption)
    st.audio(temp_wav_file_path, format='audio/wav')
    os.remove(temp_wav_file_path)

def spec_to_audio_to_st(x, spec_dir_path, sample_rate, show_griffin_lim, vocoder=None, show_in_st=True):
    # audios are in [-1, 1], making them in [0, 1]
    spec = (x.data.squeeze(0) + 1) / 2

    out = {}
    if vocoder:
        # (L,) <- wave: (1, 1, L).squeeze() <- spec: (1, F, T)
        wave_from_vocoder = vocoder(spec).squeeze().cpu().numpy()
        out['vocoder'] = wave_from_vocoder
        if show_in_st:
            show_wave_in_streamlit(wave_from_vocoder, sample_rate, 'Reconstructed Wave via MelGAN')

    if show_griffin_lim:
        spec = spec.squeeze(0).cpu().numpy()
        wave_from_griffinlim = inv_transforms(spec, Path(spec_dir_path).stem)
        out['inv_transforms'] = wave_from_griffinlim
        if show_in_st:
            show_wave_in_streamlit(wave_from_griffinlim, sample_rate, 'Reconstructed Wave via Griffin Lim')

    return out

def all_attention_to_st(attention, placeholders=None, scale_by_prior=None):
    if scale_by_prior:
        B, H, T, T = attention.shape
        # attention weight is 1/T: if we have a seq with length 3 the weights are 1/3, 1/3, and 1/3
        # making T by T matrix with zeros in the upper triangular part
        attention_uniform_prior = 1 / torch.arange(1, T+1).view(1, T, 1).repeat(B, 1, T)
        attention_uniform_prior = attention_uniform_prior.tril().view(B, 1, T, T).to(attention.device)
        attention = attention - attention_uniform_prior

    attention_agg = attention.sum(dim=1, keepdims=True)
    att_st = tensor_to_plt(attention_agg)
    # z_att_st = tensor_to_plt(z_att, flip_z_dims)
    if placeholders is None:
        return att_st
    else:
        placeholders['title_z_att'].text(f'Attention to All. {list(attention_agg.squeeze().shape)}')
        placeholders['z_att'].write(att_st)
        placeholders['title_c_att'].empty()
        placeholders['c_att'].empty()

def last_attention_to_st(attention, z_curr_step, c_length, z_permuter, c_permuter, quant_c_shape,
                         quant_z_shape, placeholders=None, flip_c_dims=None, flip_z_dims=None):
    B, H, T, T = attention.shape
    # Since the attention ignores the last (target) element, we will visualize it as 0 – padding last 2 dims
    # (B, H, T+1, T+1)
    attention = torch.nn.functional.pad(attention, pad=(0, 1, 0, 1), value=0)
    current_step = c_length + z_curr_step
    attention_at_curr_step = attention[:, :, current_step-1, :]
    # (B, H, c_length), (B, H, z_length) <-
    c_att, z_att = attention_at_curr_step[:, :, :c_length], attention_at_curr_step[:, :, c_length:]
    # aggregate through all heads H -> (B, c_length), (B, z_length)
    c_att = c_att.sum(dim=1)  # * 10
    z_att = z_att.sum(dim=1)  # * 10
    # (B, length) -> (B, 1, *2d_or_1d_code_book_shape). *shpae[2:] will take 2 elems if 2d and 1 if 1d
    c_att = c_permuter(c_att, reverse=True).reshape(B, 1, *quant_c_shape[2:])
    z_att = z_permuter(z_att, reverse=True).reshape(B, 1, *quant_z_shape[2:])
    # we don't need to flip 1d cond but we do need it for 2d input because of the spectrograms (upside-down)
    # making value in two plots in the same range
    # vmin = min(c_att.min(), z_att.min())
    # vmax = max(c_att.max(), z_att.max())
    vmin = None
    vmax = None
    c_att_st = tensor_to_plt(c_att, vmin, vmax, flip_c_dims)
    z_att_st = tensor_to_plt(z_att, vmin, vmax, flip_z_dims)
    c_att_weight = c_att.sum() / H
    z_att_weight = z_att.sum() / H
    if placeholders is None:
        return c_att_st, z_att_st
    else:
        if len(c_att.squeeze().shape) > 0:
            placeholders['title_c_att'].text(f'Attention to C. {list(c_att.squeeze().shape)}. Sum {c_att_weight:.2f}')
            placeholders['c_att'].pyplot(c_att_st)
        else:
            placeholders['c_att'].empty()
            placeholders['title_c_att'].text(f'Attention to C. Sum {c_att_weight:.2f}')
        placeholders['title_z_att'].text(f'Attention to Z. {list(z_att.squeeze().shape)}. Sum {z_att_weight:.2f}')
        placeholders['z_att'].write(z_att_st)

def get_class_preditions(x, feat_extractor, k=10):
    # use device=torch.device('cpu') to compute on cpu and save some memory
    device = x.device
    x = {'image': x.squeeze(0).cpu()}
    x = feat_extractor['transforms'](x).to(device)
    features = feat_extractor['model'](x)
    featuresdict = feat_extractor['model'].convert_features_tuple_to_dict(features)
    probs = featuresdict['logits'].softmax(dim=1)
    topk_probs, topk_targets = probs.topk(k)
    to_print = f'Spectrogram Classifier (K={k}):\n'
    for p, y in zip(topk_probs.squeeze(0).cpu().tolist(), topk_targets.squeeze(0).cpu().tolist()):
        to_print += f'\t{feat_extractor["target2label"][y]}: {p:.5f}\n'
    return to_print


def sample_conditionally(z_indices, sampling_shape, c_indices, quant_c, full_att_mat, scale_att_by_prior,
                         temperature, top_x, update_every, placeholders,
                         cond_stage_model_name, flip_z_dims, flip_c_dims, to_save_results, logdir, batch,
                         specs_key_in_batch, vocoder, feat_sampler_cfg, show_griffin_lim, feat_extractor,
                         mode):
    start_t = time.time()

    # for facehq
    # patch_size_j = 16
    # patch_size_i = 16
    patch_size_i = 5
    patch_size_j = 53

    B, D, hr_h, hr_w = sampling_shape
    # assert hr_w % patch_size_j == 0 and hr_w // patch_size_j == int(hr_w // patch_size_j)

    if mode == 'full':
        start_step = 0
    else:
        start_step = (patch_size_j // 2) * patch_size_i

    z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(z_indices.device)
    z_pred_indices[:, :start_step] = z_indices[:, :start_step]

    for step in range(start_step, hr_w * hr_h):
        i = step % hr_h
        j = step // hr_h

        i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
        j_start = min(max(0, j - (patch_size_j // 2)), hr_w - patch_size_j)
        i_end = i_start + patch_size_i
        j_end = j_start + patch_size_j

        local_i = i - i_start
        local_j = j - j_start

        patch_2d_shape = (B, D, patch_size_i, patch_size_j)

        placeholders['time'].text(f"Time: {time.time() - start_t:3.2f} seconds")
        placeholders['info'].text(
            f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})"
        )

        # TODO: faceshq – we don't need to permute the reshaped indices (1st and 2nd time)
        # slicing the possibly permuted flat sequence:
        # 1D z_pred_indices is permuted: A_flat = [1, 2, 3, 4, 5, 6, 7, 8, 9].
        # the 2D input should be: A = [[1, 4, 7], [2, 5, 8], [3, 6, 9]].
        # Therefore, after the first reshape it will be A.T = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # the last reshape flattens is back
        patch = z_pred_indices \
            .reshape(B, hr_w, hr_h) \
            .permute(0, 2, 1)[:, i_start:i_end, j_start:j_end].permute(0, 2, 1) \
            .reshape(B, patch_size_i * patch_size_j)

        # if cond_stage_model_name == 'CoordStage':
        #     cpatch = c_indices \
        #         .reshape(B, hr_w, hr_h) \
        #         .permute(0, 2, 1)[:, i_start:i_end, j_start:j_end].permute(0, 2, 1) \
        #         .reshape(B, patch_size_i * patch_size_j)
        # elif cond_stage_model_name == 'VQModel1d':
        #     cpatch = c_indices[:, j_start:j_end]
        # elif cond_stage_model_name == 'FeatsClassStage':
        #     features = quant_c['feature']
        #     if feat_sampler_cfg is None:
        #         time_step_coeff = features.shape[-1] / sampling_shape[-1]
        #         assert time_step_coeff == int(time_step_coeff), f'{features.shape}, {sampling_shape}'
        #         j_start_feats = int(j_start * time_step_coeff)
        #         j_end_feats = int(j_end * time_step_coeff)
        #     else:
        #         feat_sample_size = feat_sampler_cfg.params.feat_sample_size
        #         times_to_repeat_after_resample = feat_sampler_cfg.params.times_to_repeat_after_resample
        #         if times_to_repeat_after_resample is not None:
        #             feat_sample_size *= times_to_repeat_after_resample
        #         patches_in_z = sampling_shape[-1] // patch_size_j
        #         patches_in_c = features.shape[-1] // feat_sample_size
        #         # assert patches_in_c == patches_in_z, f'{features.shape}, {sampling_shape}'
        #         j_start_feats = j_start // patch_size_j
        #         j_end_feats = j_start + feat_sample_size
        #     cpatch = {
        #         'target': quant_c['target'],
        #         'feature': c_indices['feature'][:, :, j_start_feats:j_end_feats]
        #     }
        # elif cond_stage_model_name in ['RawFeatsStage', 'FeatClusterStage']:
        #     if feat_sampler_cfg is None:
        #         time_step_coeff = quant_c.shape[-1] / sampling_shape[-1]
        #         assert time_step_coeff == int(time_step_coeff), f'{quant_c.shape}, {sampling_shape}'
        #         j_start_feats = int(j_start * time_step_coeff)
        #         j_end_feats = int(j_end * time_step_coeff)
        #     else:
        #         feat_sample_size = feat_sampler_cfg.params.feat_sample_size
        #         times_to_repeat_after_resample = feat_sampler_cfg.params.times_to_repeat_after_resample
        #         if times_to_repeat_after_resample is not None:
        #             feat_sample_size *= times_to_repeat_after_resample
        #         patches_in_z = sampling_shape[-1] // patch_size_j
        #         patches_in_c = quant_c.shape[-1] // feat_sample_size
        #         print(patches_in_c, patches_in_z)
        #         # assert patches_in_c == patches_in_z, f'{quant_c.shape}, {sampling_shape}'
        #         j_start_feats = j_start // patch_size_j
        #         j_end_feats = j_start + feat_sample_size
        #     if cond_stage_model_name == 'FeatClusterStage':
        #         cpatch = c_indices[:, j_start_feats:j_end_feats]
        #     else:
        #         cpatch = c_indices[:, :, j_start_feats:j_end_feats]
        # elif cond_stage_model_name == 'ClassOnlyStage':
        #     cpatch = c_indices
        # else:
        #     raise NotImplementedError

        # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
        cpatch = c_indices

        if cond_stage_model_name in ['RawFeatsStage', 'ClassOnlyStage', 'FeatsClassStage']:
            logits, _, attention = model.transformer(patch[:, :-1], cpatch)
        else:
            patch = torch.cat((cpatch, patch), dim=1)
            logits, _, attention = model.transformer(patch[:, :-1])
        # remove conditioning
        logits = logits[:, -patch_size_j*patch_size_i:, :]

        local_pos_in_flat = local_j * patch_size_i + local_i
        logits = logits[:, local_pos_in_flat, :]

        logits = logits / temperature

        if top_x is not None:
            logits = model.top_k_logits(logits, top_x)
        # apply softmax to convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1)
        z_pred_indices[:, j * hr_h + i] = ix
        # print(
        #     z_pred_indices \
        #     .reshape(B, hr_w, hr_h).permute(0, 2, 1)[:, i_start:i_end, j_start:j_end].permute(0, 2, 1)
        # )
        # print(z_pred_indices.reshape(B, hr_w, hr_h).permute(0, 2, 1).permute(0, 2, 1))

        if step % update_every == 0:
            z_pred_img = model.decode_to_img(z_pred_indices, sampling_shape)
            placeholders['title_gen_spec'].text(f'Sampling {mode}. {list(z_pred_img.squeeze().shape)}')
            # fliping the spectrogram just for illustration purposes (low freqs to bottom, high - top)
            z_pred_img_st = tensor_to_plt(z_pred_img, flip_dims=flip_z_dims)
            placeholders['gen_spec'].write(z_pred_img_st)

            if full_att_mat:
                all_attention_to_st(attention, placeholders, scale_att_by_prior)
            else:
                if cond_stage_model_name == 'FeatsClassStage':
                    # 212 + 1
                    c_length = cpatch['feature'].shape[-1] + cpatch['target'].shape[-1]
                    quant_c_shape = [None, None, c_length]
                else:
                    c_length = cpatch.shape[-1]
                    quant_c_shape = quant_c.shape
                # quant_z_shape = sampling_shape
                last_attention_to_st(attention, local_pos_in_flat, c_length, model.first_stage_permuter,
                                     model.cond_stage_permuter, quant_c_shape, patch_2d_shape, placeholders,
                                     flip_c_dims, flip_z_dims)

    # quant_z_shape = sampling_shape
    z_pred_img = model.decode_to_img(z_pred_indices, sampling_shape)

    print(f'Time: {time.time() - start_t:3.2f} seconds')

    # showing the final image
    placeholders['title_gen_spec'].text(f'Sampling {mode}. {list(z_pred_img.squeeze().shape)}')
    z_pred_img_st = tensor_to_plt(z_pred_img, flip_dims=flip_z_dims)
    placeholders['gen_spec'].write(z_pred_img_st)

    if full_att_mat:
        all_attention_to_st(attention, placeholders, scale_att_by_prior)
    else:
        if cond_stage_model_name == 'FeatsClassStage':
            # 212 + 1
            c_length = cpatch['feature'].shape[-1] + cpatch['target'].shape[-1]
            quant_c_shape = [None, None, c_length]
        else:
            c_length = cpatch.shape[-1]
            quant_c_shape = quant_c.shape

        last_attention_to_st(attention, local_pos_in_flat, c_length, model.first_stage_permuter,
                             model.cond_stage_permuter, quant_c_shape, patch_2d_shape, placeholders,
                             flip_c_dims, flip_z_dims)

    topk_preds = get_class_preditions(z_pred_img, feat_extractor)
    st.text(topk_preds)

    waves = spec_to_audio_to_st(z_pred_img, config.data.params.spec_dir_path,
                                config.data.params.sample_rate, show_griffin_lim, vocoder)

    if to_save_results:
        save_results(z_pred_img_st, waves, topk_preds, logdir, batch, mode, config.data.params.sample_rate,
                     specs_key_in_batch)

    st.info('Done')


if __name__ == "__main__":
    st.sidebar.info('''
    Hi there 👋

    This is a demo for **Visually Guided Sound Generation** project 🖼️ 👉 🔉.

    [Project Page](https://v-iashin.github.io/specvqgan)
    • [Paper](https://arxiv.org/abs/2110.08791)
    • [Code](https://github.com/v-iashin/SpecVQGAN)
    • [Colab](https://colab.research.google.com/drive/1pxTIMweAKApJZ3ZFqyBee3HtMqFpnwQ0?usp=sharing)
    ''')

    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    avail_models = Path(opt.logdir).rglob('*/checkpoints')
    # 'T' is an empty model which prevents loading the first model by default
    avail_models = ['T'] + sorted([str(p.parent) for p in avail_models])
    # filtering out codebook models as we need only samplers
    avail_models = [m for m in avail_models if 'codebook' not in m]
    assert len(avail_models) > 0, f'There is no model in {opt.logdir}'
    st.sidebar.header('Select a Model')
    model_ckpt = st.sidebar.selectbox('', avail_models, 0, format_func=rename_models)
    if model_ckpt == 'T':
        st.stop()

    opt.resume = model_ckpt

    ckpt_vocoder = opt.vocoder_path
    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                template_idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt_dir = os.path.join(logdir, "checkpoints")
            ckpt_file = sorted(os.listdir(ckpt_dir))
            if len(ckpt_file) > 1:
                print(f'Warning: Found more than one checkpoint in {ckpt_dir}: {ckpt_file}')
            ckpt_file = ckpt_file[0]
            print(f'Using {ckpt_file}')
            ckpt = os.path.join(logdir, 'checkpoints', ckpt_file)
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"):
                del config["data"]
    config = OmegaConf.merge(*configs, cli)

    # determine the data folder
    if 'vggsound.VGGSound' in config.data.params.train.target:
        datapath = './data/vggsound/'
        raw_vids_dir = os.path.join(datapath, 'video')
    elif 'vas.VAS' in config.data.params.train.target:
        datapath = './data/vas/'
        raw_vids_dir = os.path.join(datapath, 'videos', '*')
    else:
        raise NotImplementedError

    # patch config. E.g. if the model is trained on another machine with different paths
    for a in ['spec_dir_path', 'rgb_feats_dir_path', 'flow_feats_dir_path']:
        if config.data.params[a] is not None:
            if 'vggsound.VGGSound' in config.data.params.train.target:
                config.data.params[a] = os.path.join(datapath, Path(config.data.params[a]).name)
            elif 'vas.VAS' in config.data.params.train.target:
                config.data.params[a] = os.path.join(datapath, 'features', '*', Path(config.data.params[a]).name)

    with st.beta_expander('Streamlit Logs'):
        dsets, model, vocoder, global_step, feat_extractor = load_model_and_dataset(
            config, ckpt, ckpt_vocoder, gpu=True, eval_mode=True
        )

    with st.beta_expander('Sampler Model Config'):
        st.text(f'Global step: {global_step}')
        st.text(f'Checkpoint: {ckpt}')
        st.json(OmegaConf.to_container(config))

    with torch.no_grad():
        if len(dsets.datasets) > 1:
            splits = sorted(dsets.datasets.keys())
            if 'vas.VAS' in config.data.params.train.target:
                # prevent loading train on demo  which results in a error in streamlit
                splits = ['validation', 'train']
            st.sidebar.header('Select Data')
            split = st.sidebar.radio('Split', splits)
            dset = dsets.datasets[split]
        else:
            dset = next(iter(dsets.datasets.values()))

        # filter dataset for available items using set intersection
        if 'vggsound.VGGSound' in config.data.params.train.target:
            avail_dataset = glob.glob(config.data.params['spec_dir_path'] + '/*_mel.npy')
            avail_dataset = sorted(list(set(avail_dataset).intersection(dset.specs_dataset.dataset)))
            avail_targets = list({dset.specs_dataset.video2target[Path(c).stem[:11]] for c in avail_dataset})
            avail_label2target = {dset.specs_dataset.target2label[t]: t for t in avail_targets}
            dset.specs_dataset.label2target = avail_label2target
            dset.specs_dataset.dataset = avail_dataset
            if hasattr(dset, 'feats_dataset'):
                avail_dataset = glob.glob(config.data.params['rgb_feats_dir_path'] + '/*.pkl')
                avail_dataset = [Path(p).stem for p in avail_dataset]
                avail_dataset = sorted(list(set(avail_dataset).intersection(dset.feats_dataset.dataset)))
                dset.feats_dataset.dataset = avail_dataset
        elif 'vas.VAS' in config.data.params.train.target:
            avail_dataset = glob.glob(config.data.params['spec_dir_path'] + '/*_mel.npy')
            avail_dataset = [os.path.join(Path(p).parent.parent.stem, Path(p).stem.replace('_mel', '')) for p in avail_dataset]
            avail_dataset = sorted(list(set(avail_dataset).intersection(dset.specs_dataset.dataset)))
            dset.specs_dataset.dataset = avail_dataset
            if hasattr(dset, 'feats_dataset'):
                avail_dataset = glob.glob(config.data.params['rgb_feats_dir_path'] + '/*.pkl')
                avail_dataset = [os.path.join(Path(p).parent.parent.stem, Path(p).stem) for p in avail_dataset]
                avail_dataset = sorted(list(set(avail_dataset).intersection(dset.feats_dataset.dataset)))
                dset.feats_dataset.dataset = avail_dataset

        if len(dset) == 0:
            st.sidebar.info('There are no samples for this split. Please select another split.')
            st.stop()

        select_specific_class = st.sidebar.checkbox('Select Specific Class...', value=False)

        # add available classes
        if select_specific_class:
            labels = dset.specs_dataset.label2target.keys()
            label_choice = st.sidebar.selectbox('Select a Class', sorted(labels))
            # filter dataset for observations belonging to a specific class
            label2target = dset.specs_dataset.label2target
            if 'vggsound.VGGSound' in config.data.params.train.target:
                video2target = dset.specs_dataset.video2target
                paths = dset.specs_dataset.dataset
                filter_paths = [c for c in paths if video2target[Path(c).stem[:11]] == label2target[label_choice]]
                dset.specs_dataset.dataset = filter_paths
                # if we have another first stage we need to do something extra
                if hasattr(dset, 'feats_dataset'):
                    paths_feats = dset.feats_dataset.dataset
                    filter_paths_feats = [c for c in paths_feats if video2target[Path(c).stem[:11]] == label2target[label_choice]]
                    dset.feats_dataset.dataset = filter_paths_feats
            elif 'vas.VAS' in config.data.params.train.target:
                paths = dset.specs_dataset.dataset
                filter_paths = [c for c in paths if c.startswith(label_choice)]
                dset.specs_dataset.dataset = filter_paths
                # if we have another first stage we need to do something extra
                if hasattr(dset, 'feats_dataset'):
                    paths_feats = dset.feats_dataset.dataset
                    filter_paths_feats = [c for c in paths_feats if c.startswith(label_choice)]
                    dset.feats_dataset.dataset = filter_paths_feats

        batch_size = 1
        start_index = st.sidebar.number_input(f'Example Index in the Dataset [0, {len(dset)-1}]',
                                              value=0, min_value=0, max_value=len(dset)-batch_size)
        indices = list(range(start_index, start_index+batch_size))

        batch = default_collate([dset[i] for i in indices])

        if select_specific_class:
            # restoring original dataset because we cached the dataset class and filtered for one class.
            # Next time, the filtered dataset will be filtered again which empties the dataset.
            dset.specs_dataset.dataset = paths
            # if we have another first stage we need to do something extra
            if hasattr(dset, 'feats_dataset'):
                dset.feats_dataset.dataset = paths_feats

        feat_sampler_cfg = dset.condition_dataset_cfg.feat_sampler_cfg
        cond_stage_model_name = model.cond_stage_model.__class__.__name__
        transformer_model_name = model.transformer.__class__.__name__

        if (cond_stage_model_name in ['VQModel1d', 'FeatClusterStage']
            or transformer_model_name in ['GPTFeats', 'GPTFeatsClass']):
            specs_key_in_batch = 'file_path_specs_'
            flip_c_dims = None
        elif transformer_model_name == 'GPTClass':
            specs_key_in_batch = 'file_path_'
            flip_c_dims = None
        else:
            specs_key_in_batch = 'file_path_'
            flip_c_dims = (2,)
        flip_z_dims = (2,)

        st.text('')
        with st.beta_expander(f'Original Video. Class: {batch["label"]}.'):
            vid_fname = Path(batch[specs_key_in_batch][0]).name.replace('_mel.npy', '.mp4')
            st.text(f'Video file name: {vid_fname}')
            if 'vggsound.VGGSound' in config.data.params.train.target:
                video_file = open(os.path.join(raw_vids_dir, vid_fname), 'rb').read()
            elif 'vas.VAS' in config.data.params.train.target:
                cls = batch['label'][0]
                video_file = open(os.path.join(raw_vids_dir.replace('*', cls), vid_fname), 'rb').read()
            st.video(video_file, format='video/mp4')

        x = model.get_input(model.first_stage_key, batch).to(model.device)
        c = model.get_input(model.cond_stage_key, batch)
        if isinstance(c, dict):
            c = {k: v.to(model.device) for k, v in c.items()}
        else:
            c = c.to(model.device)

        quant_z, z_indices = model.encode_to_z(x)
        quant_c, c_indices = model.encode_to_c(c)

        xrec = model.first_stage_model.decode(quant_z)
        crec = model.cond_stage_model.decode(quant_c)

        if transformer_model_name == 'GPTFeatsClass':
            orig_cond_shape = c['feature'].squeeze().shape
            rec_cond_shape = crec["feature"].squeeze().shape
        else:
            orig_cond_shape = c.squeeze().shape
            rec_cond_shape = crec.squeeze().shape

        st.text('')
        with st.beta_expander(f'Conditioning {list(orig_cond_shape)}'):
            if transformer_model_name == 'GPTClass':
                st.write(batch['label'])
            elif transformer_model_name == 'GPTFeatsClass':
                st.write(batch['label'])
                st.write(tensor_to_plt(c['feature'], flip_dims=flip_c_dims))
            else:
                st.write(tensor_to_plt(c, flip_dims=flip_c_dims))
        # with st.beta_expander(f'Conditioning Reconstruction {list(rec_cond_shape)}'):
        #     if transformer_model_name == 'GPTClass':
        #         st.write(batch['label'])
        #     elif transformer_model_name == 'GPTFeatsClass':
        #         st.write(batch['label'])
        #         st.write(tensor_to_plt(crec['feature'], flip_dims=flip_c_dims))
        #     else:
        #         st.write(tensor_to_plt(crec, flip_dims=flip_c_dims))

        st.sidebar.header('Results Handling')
        update_every = st.sidebar.number_input('Display Result Every ... Step', value=3)
        show_griffin_lim = st.sidebar.checkbox(
            'Also Show Griffin-Lim', value=False,
            help='Show spectrogram reconstruction from Griffin-Lim algorithm along the pre-trained vocoder')
        to_save_results = st.sidebar.checkbox('Save Results', value=True)

        st.text('')
        with st.beta_expander(f'Input {list(x.squeeze().shape)}'):
            st.write(tensor_to_plt(x, flip_dims=flip_z_dims))
            topk_results = get_class_preditions(x, feat_extractor)
            st.text(topk_results)
            if st.button('Get Audio (Input)'):
                spec_to_audio_to_st(x, config.data.params.spec_dir_path,
                                    config.data.params.sample_rate, show_griffin_lim, vocoder)
        with st.beta_expander(f'Input Reconstruction from SpecVQGAN {list(xrec.squeeze().shape)}', expanded=True):
            st.write(tensor_to_plt(xrec, flip_dims=flip_z_dims))
            topk_results = get_class_preditions(xrec, feat_extractor)
            st.text(topk_results)
            if st.button('Get Audio (Input Reconstruction)'):
                spec_to_audio_to_st(xrec, config.data.params.spec_dir_path,
                                    config.data.params.sample_rate, show_griffin_lim, vocoder)

        st.sidebar.header('Sampling Parameters')
        temperature = st.sidebar.number_input(
            'Softmax Temperature', value=1.0,
            help='$T$ in $\exp(x_i/T) / \Sigma_j \exp(x_j/T)$'
        )
        top_x = st.sidebar.number_input(
            'Top X', value=config.model.params.first_stage_config.params.n_embed // 2,
            help='Cuts sampling space of the next token to Top $X$ highest probability tokens. '
                 + 'It increases diversity of samples but at the cost of relevance. '
                 + 'As a rule of thumb, use `X = |codebook| // 2`.'
        )
        W_scale = st.sidebar.number_input(
            'Temporal Scale', value=1, min_value=1,
            help='The output length is `temporal_scale * 9.8 seconds`.')
        sample_half = st.sidebar.checkbox(
            'Prime with GT Tokens', value=False,
            help='If checked, the first half of the tokens will be taken from the ground truth audio'
            + ' codebook representation and sampling will continue this sequence.')
        full_att_mat = st.sidebar.checkbox(
            'Show Full Attention Matrix', value=False,
            help='The attention will be shown for each time stamp instead of only the current one.')
        if full_att_mat:
            scale_att_by_prior = st.sidebar.checkbox(
                'Subtract Prior from Attention', value=True,
                help='If checked, subtracts $1/S$ from each attention weight, where $S$ is number of'
                + ' previous tokens. For example, $[2/3, 1/6, 1/6]~–~[1/3, 1/3, 1/3] = [1/3, -1/6, -1/6]$')
        else:
            scale_att_by_prior = False

        st.header('Sampling Results:')

        # dummy outputs just to reserver some space
        placeholders = {
            'info': st.text('Step: (?,?) | Local: (?,?) | Crop: (?:?,?:?)'),
            'time': st.text('Time: ?'),
            'mode': st.text('Mode: ?'),
            'title_c_att': st.text('Attention to C.'),
            'c_att': st.pyplot(tensor_to_plt(torch.zeros_like(x))),
            'title_z_att': st.text('Attention to Z.'),
            'z_att': st.pyplot(tensor_to_plt(torch.zeros_like(x))),
            'title_gen_spec': st.text('Generated sample'),
            'gen_spec': st.pyplot(tensor_to_plt(torch.zeros_like(x))),
            'title_rec_audio': st.text('Reconstructed Audio of the Generated Sample'),
        }

        sampling_shape = list(quant_z.shape)
        # hr_w * w_scale
        sampling_shape[3] *= W_scale

        if st.sidebar.button('Start Sampling'):
            mode = 'half' if sample_half else 'full'
            sample_conditionally(
                z_indices,
                sampling_shape,
                c_indices,
                quant_c,
                full_att_mat,
                scale_att_by_prior,
                temperature,
                top_x,
                update_every,
                placeholders,
                cond_stage_model_name,
                flip_z_dims,
                flip_c_dims,
                to_save_results,
                logdir,
                batch,
                specs_key_in_batch,
                vocoder,
                feat_sampler_cfg,
                show_griffin_lim,
                feat_extractor,
                mode
            )
