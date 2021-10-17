import datetime
import os
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import sys
from tqdm import tqdm
import soundfile

sys.path.insert(0, '.')  # nopep8
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from train import instantiate_from_config
from copy import deepcopy

def save_config(cfg):
    cfg_to_save = deepcopy(cfg)
    # remove sensitive info
    for key in ['config_sampler', 'device_ids', 'nodes', 'node_id', 'master_port', 'master_addr']:
        if key in cfg_to_save.sampler:
            cfg_to_save.sampler.pop(key)
    # make a path, dir and save the config
    save_dir = Path(cfg_to_save.sampler.model_logdir) / f'samples_{cfg.sampler.now}'
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg_to_save, save_dir / 'full_sampling_cfg.yaml')

def load_and_save_config():
    # cli arguments and configs
    config_cli = OmegaConf.from_cli()
    config_sampler = OmegaConf.load(config_cli.sampler.config_sampler)
    cfg_cli_sampler = OmegaConf.merge(config_sampler, config_cli)
    model_logdir = Path(cfg_cli_sampler.sampler.model_logdir)
    assert Path(model_logdir).exists(), f'Specified: {model_logdir}'
    # load configs from the model specified in model_logdir
    # there could be several *-project.yaml configs e.g. because the experiment was resumed
    configs_model_base = Path(model_logdir).glob('configs/*-project.yaml')
    configs_lightning_base = Path(model_logdir).glob('configs/*-lightning.yaml')
    configs_model_base = [OmegaConf.load(p) for p in configs_model_base]
    configs_lightning_base = [OmegaConf.load(p) for p in configs_lightning_base]
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(*configs_model_base, *configs_lightning_base, cfg_cli_sampler)
    # patch cfg. E.g. if the model is trained on another machine with different paths
    for a in ['spec_dir_path', 'rgb_feats_dir_path', 'flow_feats_dir_path']:
        # TODO: `config_cli.data` is a bit ambigiuous because we can pass some other fields besides paths in data
        if config_cli.data is None:
            if cfg.data.params[a] is not None:
                if 'vggsound.VGGSound' in cfg.data.params.train.target:
                    cfg.data.params[a] = os.path.join('./data/vggsound/', Path(cfg.data.params[a]).name)
                elif 'vas.VAS' in cfg.data.params.train.target:
                    cfg.data.params[a] = os.path.join('./data/vas/features/*', Path(cfg.data.params[a]).name)

    # save the config
    save_config(cfg)
    return cfg

def make_folder_for_samples(cfg):
    assert Path(cfg.sampler.model_logdir).exists()
    dataset_name = cfg.data.params.train.target.split('.')[-1]
    if 'VGGSound' in dataset_name:
        dataset_name = 'VGGSound'
    elif 'VAS' in dataset_name:
        dataset_name = 'VAS'
    else:
        raise NotImplementedError
    # init a folder with samples
    samples_split_dirs = {}
    for split in cfg.sampler.splits:
        samples_split_dirs[split] = Path(cfg.sampler.model_logdir) / f'samples_{cfg.sampler.now}' / f'{dataset_name}_{split}'
        os.makedirs(samples_split_dirs[split], exist_ok=True)
    return samples_split_dirs


def load_model_and_dataloaders(cfg, device, is_ddp=False):
    # find the checkpoint path
    # e.g. checkpoints have names `epoch_000012.ckpt`
    if (Path(cfg.sampler.model_logdir) / 'checkpoints/last.ckpt').exists():
        ckpt_model = Path(cfg.sampler.model_logdir) / 'checkpoints/last.ckpt'
    else:
        ckpt_model = sorted(Path(cfg.sampler.model_logdir).glob('checkpoints/*.ckpt'))[-1]
    # assert not (Path(cfg.sampler.model_logdir) / 'checkpoints/last.ckpt').exists()
    print(f'Going to use the checkpoint from {ckpt_model}')

    # get data
    dsets = instantiate_from_config(cfg.data)
    dsets.prepare_data()
    dsets.setup()
    dsets.datasets = {split: dset for split, dset in dsets.datasets.items() if split in cfg.sampler.splits}

    # loading the vocoder
    # ckpt_vocoder = cfg.lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder
    # if ckpt_vocoder:
    #     vocoder = load_vocoder(ckpt_vocoder, eval_mode=True)['model'].to('cuda')
    vocoder = None

    # now load the specified checkpoint
    if ckpt_model:
        pl_sd = torch.load(ckpt_model, map_location='cpu')
        global_step = pl_sd['global_step']
    else:
        pl_sd = {'state_dict': None}
        global_step = None

    if 'ckpt_path' in cfg.model.params:
        cfg.model.params.ckpt_path = None
    if 'downsample_cond_size' in cfg.model.params:
        cfg.model.params.downsample_cond_size = -1
        cfg.model.params['downsample_cond_factor'] = 0.5
    try:
        if 'ckpt_path' in cfg.model.params.first_stage_config.params:
            cfg.model.params.first_stage_config.params.ckpt_path = None
        if 'ckpt_path' in cfg.model.params.cond_stage_config.params:
            cfg.model.params.cond_stage_config.params.ckpt_path = None
    except Exception:
        pass
    model = instantiate_from_config(cfg.model)
    if pl_sd['state_dict'] is not None:
        missing, unexpected = model.load_state_dict(pl_sd['state_dict'], strict=False)
        print(f'Missing fields: {missing}')
        print(f'Unexpected fields: {unexpected}')
    model.to(device)
    model.eval()

    assert not model.training, 'The model is in "training" mode'

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    # defining data loaders in a dict
    dloaders = {}
    for split, dset in dsets.datasets.items():
        if is_ddp:
            sampler = DistributedSampler(dset, dist.get_world_size(), dist.get_rank(), shuffle=False)
            num_workers = 0
        else:
            sampler = None
            num_workers = cfg.sampler.num_workers
        dloaders[split] = DataLoader(dset, cfg.sampler.batch_size, sampler=sampler, num_workers=num_workers,
                                     pin_memory=True, drop_last=False)
    return dloaders, model, vocoder, global_step

def sample_spectrogram(cfg, model, batch):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    with torch.no_grad():
        # if we use VQGAN (for ablation study only), sampling is pretty straight-forward
        if 'VQModel' in model.__class__.__name__:
            return model.log_images(batch)['reconstructions']

        x = model.get_input(model.first_stage_key, batch).to(model.device)
        # c = model.get_input(model.cond_stage_key, batch).to(model.device)
        c = model.get_input(model.cond_stage_key, batch)
        if isinstance(c, dict):
            c = {k: v.to(model.device) for k, v in c.items()}
        else:
            c = c.to(model.device)

        # applying pre-trained VQGAN encoder for input (x) and, optionally, conditioning
        quant_z, z_indices = model.encode_to_z(x)
        quant_c, c_indices = model.encode_to_c(c)

        B, _, hr_h, hr_w = quant_z.shape

        if cfg.sampler.sampling_mode == 'nopix':
            start_step = 0
        else:
            start_step = (hr_w // 2) * hr_h

        z_pred_indices = torch.zeros_like(z_indices)
        z_pred_indices[:, :start_step] = z_indices[:, :start_step]

        for step in range(start_step, hr_w * hr_h):
            i = step % hr_h
            j = step // hr_h

            patch = z_pred_indices
            cpatch = c_indices

            if model.cond_stage_model.__class__.__name__ in ['RawFeatsStage', 'ClassOnlyStage', 'FeatsClassStage']:
                logits, _, attention = model.transformer(patch[:, :-1], cpatch)
            else:
                patch = torch.cat((cpatch, patch), dim=1)
                logits, _, attention = model.transformer(patch[:, :-1])

            # remove conditioning
            logits = logits[:, -hr_w*hr_h:, :]

            local_pos_in_flat = j * hr_h + i
            logits = logits[:, local_pos_in_flat, :]

            logits = logits / cfg.sampler.temperature

            if cfg.sampler.top_k is not None:
                logits = model.top_k_logits(logits, cfg.sampler.top_k)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if cfg.sampler.sample_next_tok_from_pred_dist:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            # slicing in brackets means 'leave this dimension without squeezing'
            z_pred_indices[:, [j * hr_h + i]] = ix
            # print(z_pred_indices.reshape(B, hr_w, hr_h))

        # applying pre-trained VQGAN decoder
        z_pred_img = model.decode_to_img(z_pred_indices, quant_z.shape)

        return z_pred_img


def save_specs(cfg, specs, samples_split_dirs, model, batch, split, sample_id, vocoder):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # model_ddp = model
        model = model.module

    if 'VQModel' in model.__class__.__name__:
        specs_key = 'file_path_'
    else:
        cond_stage_model_name = model.cond_stage_model.__class__.__name__
        transformer_model_name = model.transformer.__class__.__name__
        if (cond_stage_model_name in ['VQModel1d', 'FeatClusterStage']
            or transformer_model_name in ['GPTFeats', 'GPTFeatsClass']):
            specs_key = 'file_path_specs_'
        else:
            specs_key = 'file_path_'

    dataset_name = cfg.data.params.train.target.split('.')[-1]
    if 'VGGSound' in dataset_name:
        extract_vidname_fn = lambda x: Path(x).name.replace('_mel.npy', '')
    elif 'VAS' in dataset_name:
        extract_vidname_fn = lambda x: Path(x).stem
    else:
        raise NotImplementedError

    # iterating inside a batch
    for i, spec in enumerate(specs):
        class_foldername = f'cls_{batch["target"][i]}'
        vidname = extract_vidname_fn(batch[specs_key][i])
        save_path = samples_split_dirs[split] / class_foldername
        os.makedirs(save_path, exist_ok=True)
        # spec = torch.clamp(spec, -1., 1.)
        spec = spec.squeeze(0).cpu().numpy()
        # audios are in [-1, 1], making them in [0, 1]
        spec = (spec + 1) / 2
        np.save(save_path / f'{vidname}_sample_{sample_id}.npy', spec)
        if vocoder is not None:
            wave_from_vocoder = vocoder(torch.from_numpy(spec).unsqueeze(0).to('cuda')).cpu().squeeze().detach().numpy()
            soundfile.write(save_path / f'{vidname}_sample_{sample_id}.wav', wave_from_vocoder, 22050, 'PCM_24')


def sample(gpu_id, cfg, samples_split_dirs, is_ddp):

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    dataloaders, model, vocoder, _ = load_model_and_dataloaders(cfg, device, is_ddp)

    if isinstance(cfg.sampler.samples_per_video, int) and cfg.sampler.samples_per_video > 0:
        samples_per_video = cfg.sampler.samples_per_video
        print(f'USING {samples_per_video} samples per video')
    else:
        samples_per_video = 1

    for split, dataloader in dataloaders.items():
        for batch in tqdm(dataloader):
            for sample_id in range(samples_per_video):
                specs = sample_spectrogram(cfg, model, batch)
                save_specs(cfg, specs, samples_split_dirs, model, batch, split, sample_id, vocoder)

def main():
    torch.manual_seed(0)
    local_rank = os.environ.get('LOCAL_RANK')

    cfg = load_and_save_config()
    samples_split_dirs = make_folder_for_samples(cfg)

    if local_rank is not None:
        is_ddp = True
        local_rank = int(local_rank)
        # 300s is a timeout for other worker to check out when the first one reached the barrier
        dist.init_process_group(cfg.sampler.get('dist_backend', 'nccl'), 'env://', datetime.timedelta(0, 300))
        print(f'WORLDSIZE {dist.get_world_size()} – RANK {dist.get_rank()}')
        if dist.get_rank() == 0:
            print('MASTER:', os.environ['MASTER_ADDR'], ':', os.environ['MASTER_PORT'])
            print(OmegaConf.to_yaml(cfg))
    else:
        is_ddp = False
        print(OmegaConf.to_yaml(cfg))
        local_rank = 0

    sample(local_rank, cfg, samples_split_dirs, is_ddp)


if __name__ == '__main__':
    main()
