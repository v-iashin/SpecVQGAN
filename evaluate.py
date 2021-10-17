'''
Adapted from `https://github.com/toshas/torch-fidelity`.
Modified by Vladimir Iashin, 2021.
'''
import itertools
import multiprocessing
import os
from pathlib import Path
from specvqgan.util import get_ckpt_path

import torch
import torch.distributed as dist
import torchvision
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from evaluation.metrics.fid import calculate_fid
from evaluation.metrics.isc import calculate_isc
from evaluation.metrics.kid import calculate_kid
from evaluation.metrics.kl import calculate_kl
from train import instantiate_from_config


def patch_cfg_for_new_paths(nested_dict, patch):
    if patch is None:
        print('Nothing to patch')
        return nested_dict
    if not isinstance(nested_dict, (dict, DictConfig)):
        # print('Ignoring the in patching', nested_dict, type(nested_dict))
        return nested_dict
    for key, value in nested_dict.items():
        if isinstance(value, (dict, DictConfig)):
            nested_dict[key] = patch_cfg_for_new_paths(value, patch)
        elif isinstance(value, (list, ListConfig)):
            for i, element in enumerate(value):
                value[i] = patch_cfg_for_new_paths(element, patch)
        else:
            if key in patch:
                print(f'Patched {key}: {nested_dict[key]} --> {patch[key]}')
                nested_dict[key] = patch.get(key, nested_dict[key])
    return nested_dict

def get_dataset_class(dataset_cfg):
    if 'target' in dataset_cfg:
        dataset_class = instantiate_from_config(dataset_cfg)
    elif 'path_to_exp' in dataset_cfg:
        dataset_class = instantiate_from_config(dataset_cfg.exp_dataset)
        dataset_class.prepare_data()
        dataset_class.setup()
        dataset_class = dataset_class.datasets[dataset_cfg.key]
    else:
        raise NotImplementedError

    return dataset_class


def get_featuresdict(feat_extractor, device, dataset_cfg, is_ddp, batch_size, save_cpu_ram):
    input = get_dataset_class(dataset_cfg)
    # for debugging
    # input.specs_dataset.dataset = input.specs_dataset.dataset[:1000]
    # input.feats_dataset.dataset = input.feats_dataset.dataset[:1000]
    batch_size = min(batch_size, len(input))

    if dataset_cfg.transform_dset_out_to_inception_in is not None:
        transforms = [instantiate_from_config(c) for c in dataset_cfg.transform_dset_out_to_inception_in]
    else:
        transforms = [lambda x: x]
    transforms = torchvision.transforms.Compose(transforms)

    if is_ddp:
        sampler = DistributedSampler(input, dist.get_world_size(), dist.get_rank(), shuffle=False)
        num_workers = 0
    else:
        sampler = None
        num_workers = 0 if save_cpu_ram else min(8, 2 * multiprocessing.cpu_count())

    dataloader = DataLoader(
        input,
        batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device != 'cpu',
    )

    out = None
    out_meta = None

    for batch in tqdm(dataloader):
        # saving batch meta so that we could merge predictions from both datasets when
        # pair-wise KL is calculated
        # comenting out target and label, because those are asigned by folder name, not original labels
        metadict = {
            # 'target': batch['target'].cpu().tolist(),
            # 'label': batch['label'],
            'file_path_': batch.get('file_path_', batch.get('file_path_specs_')),
        }
        batch = transforms(batch)
        batch = batch.to(device)

        with torch.no_grad():
            features = feat_extractor(batch)

        featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    # we need to send the results from all ranks to one of them (gather). Otherwise, the metric is calculated
    # only on the subset of data that one worker had
    if is_ddp:
        for k, v in out.items():
            gather_out = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out, v)
            out[k] = torch.cat(gather_out)
        for k, v in out_meta.items():
            gather_out_meta = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out_meta, v)
            # just flattens the list
            out_meta[k] = list(itertools.chain(*gather_out_meta))
    # merging both dicts key-wise
    out = {**out, **out_meta}
    return out

def main():
    torch.manual_seed(0)
    local_rank = os.environ.get('LOCAL_RANK')

    cfg_cli = OmegaConf.from_cli()
    cfg_eval = OmegaConf.load(cfg_cli.config)

    # the latter arguments are prioritized
    for dataset in ['input1', 'input2']:
        cli_dataset_cfg = cfg_cli[dataset]
        # first I check if the path_to_exp is specified in CLI args
        if cli_dataset_cfg is not None:
            if cli_dataset_cfg.path_to_exp is not None:
                cfg_paths = Path(cli_dataset_cfg.path_to_exp).glob('configs/*-project.yaml')
                cfgs_dataset = [OmegaConf.load(p) for p in sorted(list(cfg_paths))]
                cfg_eval[dataset].exp_dataset = OmegaConf.merge(*cfgs_dataset).data
            else:
                assert cli_dataset_cfg.params.root is not None, 'path_to_exp or root should be specified'
        # if not specified in CLI, I will check the default config
        else:
            eval_dataset_cfg = cfg_eval[dataset]
            if eval_dataset_cfg.path_to_exp is not None:
                cfg_paths = Path(eval_dataset_cfg.path_to_exp).glob('configs/*-project.yaml')
                cfgs_dataset = [OmegaConf.load(p) for p in sorted(list(cfg_paths))]
                cfg_eval[dataset].exp_dataset = OmegaConf.merge(*cfgs_dataset).data
            else:
                assert eval_dataset_cfg.params.root is not None, 'path_to_exp or root should be specified'
    cfg = OmegaConf.merge(cfg_eval, cfg_cli)
    cfg = patch_cfg_for_new_paths(cfg, cfg.patch)

    assert cfg.have_isc or cfg.have_fid or cfg.have_kid, 'Select at least one metric'
    assert (not cfg.have_fid) and (not cfg.have_kid) or cfg.input2 is not None, 'Two inputs are required'

    if local_rank is not None:
        is_ddp = True
        local_rank = int(local_rank)
        dist.init_process_group(cfg.get('dist_backend', 'nccl'), 'env://')
        print(f'WORLDSIZE {dist.get_world_size()} – RANK {dist.get_rank()}')
        if dist.get_rank() == 0:
            print('MASTER:', os.environ['MASTER_ADDR'], ':', os.environ['MASTER_PORT'])
            print(OmegaConf.to_yaml(cfg))
    else:
        is_ddp = False
        local_rank = cfg.device[-1]  # extracting last elements from e.g. 'cuda:0'[-1]
        print(OmegaConf.to_yaml(cfg))

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # downloading the checkpoint for melception
    get_ckpt_path('melception', 'evaluation/logs/21-05-10T09-28-40')

    feat_extractor = instantiate_from_config(cfg.feature_extractor)
    feat_extractor.eval()
    feat_extractor.to(device)

    out = {}

    print('Extracting features from input_1')
    featuresdict_1 = get_featuresdict(feat_extractor, device, cfg.input1, is_ddp, **cfg.extraction_cfg)

    featuresdict_2 = None
    if cfg.input2 not in ['None', None, 'null', 'none']:
        print('Extracting features from input_2')
        featuresdict_2 = get_featuresdict(feat_extractor, device, cfg.input2, is_ddp, **cfg.extraction_cfg)

    # pickle.dump(featuresdict_1, open('feats1.pkl', 'wb'))
    # pickle.dump(featuresdict_2, open('feats2.pkl', 'wb'))

    if cfg.have_kl:
        metric_kl = calculate_kl(featuresdict_1, featuresdict_2, **cfg.kl_cfg)
        out.update(metric_kl)
    if cfg.have_isc:
        metric_isc = calculate_isc(featuresdict_1, **cfg.isc_cfg)
        out.update(metric_isc)
    if cfg.have_fid:
        metric_fid = calculate_fid(featuresdict_1, featuresdict_2, **cfg.fid_cfg)
        out.update(metric_fid)
    if cfg.have_kid:
        metric_kid = calculate_kid(featuresdict_1, featuresdict_2, **cfg.kid_cfg)
        out.update(metric_kid)

    print('\n'.join((f'{k}: {v:.7f}' for k, v in out.items())))

    # just pretty printing of the results, nothing more
    print(
        f'{cfg.input1.get("path_to_exp", Path(cfg.input1.params.root).parent.stem)}:',
        f'KL: {out.get("kullback_leibler_divergence", float("nan")):8.5f};',
        f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
        f'FID: {out.get("frechet_inception_distance", float("nan")):8.5f};',
        f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
        f'({out.get("kernel_inception_distance_std", float("nan")):.5f})'
    )


if __name__ == '__main__':
    main()
