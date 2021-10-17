import os
from glob import glob

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import instantiate_from_config


class FeatClusterStage(object):

    def __init__(self, num_clusters=None, cached_kmeans_path=None, feats_dataset_config=None, num_workers=None):
        if cached_kmeans_path is not None and os.path.exists(cached_kmeans_path):
            print(f'Precalculated Clusterer already exists, loading from {cached_kmeans_path}')
            self.clusterer = joblib.load(cached_kmeans_path)
        elif feats_dataset_config is not None:
            self.clusterer = self.load_or_precalculate_kmeans(num_clusters, feats_dataset_config, num_workers)
        else:
            raise Exception('Neither `feats_dataset_config` nor `cached_kmeans_path` are defined')

    def eval(self):
        return self

    def encode(self, c):
        # c_quant: cluster centers, c_ind: cluster index

        B, D, T = c.shape
        # (B*T, D) <- (B, T, D) <- (B, D, T)
        c_flat = c.permute(0, 2, 1).view(B*T, D).cpu().numpy()

        c_ind = self.clusterer.predict(c_flat)
        c_quant = self.clusterer.cluster_centers_[c_ind]

        c_ind = torch.from_numpy(c_ind).to(c.device)
        c_quant = torch.from_numpy(c_quant).to(c.device)

        c_ind = c_ind.long().unsqueeze(-1)
        c_quant = c_quant.view(B, T, D).permute(0, 2, 1)

        info = None, None, c_ind
        # (B, D, T), (), ((), (768, 1024), (768, 1))
        return c_quant, None, info

    def decode(self, c):
        return c

    def get_input(self, batch, k):
        x = batch[k]
        x = x.permute(0, 2, 1).to(memory_format=torch.contiguous_format)
        return x.float()

    def load_or_precalculate_kmeans(self, num_clusters, dataset_cfg, num_workers):
        print(f'Calculating clustering K={num_clusters}')
        batch_size = 64
        dataset_name = dataset_cfg.target.split('.')[-1]
        cached_path = os.path.join('./specvqgan/modules/misc/', f'kmeans_K{num_clusters}_{dataset_name}.sklearn')
        feat_depth = dataset_cfg.params.condition_dataset_cfg.feat_depth
        feat_crop_len = dataset_cfg.params.condition_dataset_cfg.feat_crop_len

        feat_loading_dset = instantiate_from_config(dataset_cfg)
        feat_loading_dset = DataLoader(feat_loading_dset, batch_size, num_workers=num_workers, shuffle=True)

        clusterer = MiniBatchKMeans(num_clusters, batch_size=batch_size*feat_crop_len, random_state=0)

        for item in tqdm(feat_loading_dset):
            batch = item['feature'].reshape(-1, feat_depth).float().numpy()
            clusterer.partial_fit(batch)

        joblib.dump(clusterer, cached_path)
        print(f'Saved the calculated Clusterer @ {cached_path}')
        return clusterer


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load('./configs/vggsound_featcluster_transformer.yaml')
    config.model.params.first_stage_config.params.ckpt_path = './logs/2021-05-19T22-16-54_vggsound_specs_vqgan/checkpoints/epoch_39.ckpt'
    model = instantiate_from_config(config.model.params.cond_stage_config)
    print(model)
