import argparse
import datetime
import glob
import importlib
import os
import signal
import sys
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile
import torch
import torchvision
import yaml
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from feature_extraction.extract_mel_spectrogram import inv_transforms
from vocoder.modules import Generator


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='postfix for logdir',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='resume from logdir or checkpoint in logdir',
    )
    parser.add_argument(
        '-b',
        '--base',
        nargs='*',
        metavar='base_config.yaml',
        help='paths to base configs. Loaded from left-to-right. '
        'Parameters can be overwritten or added with command-line options of the form `--key value`.',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='train',
    )
    parser.add_argument(
        '--no-test',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='disable test',
    )
    parser.add_argument('-p', '--project', help='name of new or path to existing project')
    parser.add_argument(
        '-d',
        '--debug',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='enable post-mortem debugging',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f',
        '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class WrappedDataset(Dataset):
    '''Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset'''
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn,
                          shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    def _test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

class SpectrogramDataModuleFromConfig(DataModuleFromConfig):
    '''avoiding duplication of hyper-parameters in the config by gross patching here '''
    def __init__(self, batch_size, num_workers, spec_dir_path=None,
                 sample_rate=None, mel_num=None, spec_len=None, spec_crop_len=None,
                 random_crop=None, train=None, validation=None, test=None, wrap=False):
        specs_dataset_cfg = {
            # 'spec_dir_name': Path(spec_dir_path).name,
            'spec_dir_path': spec_dir_path,
            'random_crop': random_crop,
            # 'sample_rate': sample_rate,
            'mel_num': mel_num,
            'spec_len': spec_len,
            'spec_crop_len': spec_crop_len,
        }
        for name, split in {'train': train, 'validation': validation, 'test': test}.items():
            if split is not None:
                split.params.specs_dataset_cfg = specs_dataset_cfg
        super().__init__(batch_size, train, validation, test, wrap, num_workers)


class ConditionedSpectrogramDataModuleFromConfig(DataModuleFromConfig):
    '''avoiding duplication of hyper-parameters in the config by gross patching here '''
    def __init__(self, batch_size, num_workers, spec_dir_path=None, rgb_feats_dir_path=None,
                 flow_feats_dir_path=None, sample_rate=None, mel_num=None, spec_len=None, spec_crop_len=None,
                 random_crop=None,  replace_feats_with_random=None,
                 feat_depth=None, feat_len=None, feat_crop_len=None, crop_coord=None,
                 for_which_class=None, feat_sampler_cfg=None, train=None, validation=None, test=None,
                 wrap=False):
        specs_dataset_cfg = {
            # 'spec_dir_name': Path(spec_dir_path).name,
            'spec_dir_path': spec_dir_path,
            'random_crop': random_crop,
            # 'sample_rate': sample_rate,
            'mel_num': mel_num,
            'spec_len': spec_len,
            'spec_crop_len': spec_crop_len,
            'crop_coord': crop_coord,
            'for_which_class': for_which_class,
        }
        condition_dataset_cfg = {
            'rgb_feats_dir_path': rgb_feats_dir_path,
            'flow_feats_dir_path': flow_feats_dir_path,
            'feat_depth': feat_depth,
            'feat_len': feat_len,
            'feat_crop_len': feat_crop_len,
            'random_crop': random_crop,
            'for_which_class': for_which_class,
            'feat_sampler_cfg': feat_sampler_cfg,
            'replace_feats_with_random': replace_feats_with_random,
        }
        for name, split in {'train': train, 'validation': validation, 'test': test}.items():
            if split is not None:

                if (split.target.split('.')[-1].startswith('VGGSoundSpecsCondOnFeats') \
                   or split.target.split('.')[-1].startswith('VASSpecsCondOnFeats')):
                    split_path = split.params.condition_dataset_cfg.split_path
                    condition_dataset_cfg['split_path'] = split_path
                split.params.condition_dataset_cfg = condition_dataset_cfg
                split.params.specs_dataset_cfg = specs_dataset_cfg

        super().__init__(batch_size, train, validation, test, wrap, num_workers)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print('Project config')
            print(self.config.pretty())
            OmegaConf.save(self.config, os.path.join(self.cfgdir, '{}-project.yaml'.format(self.now)))

            print('Lightning config')
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}),
                           os.path.join(self.cfgdir, '{}-lightning.yaml'.format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, 'child_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class VocoderMelGan(object):

    def __init__(self, ckpt_vocoder):
        ckpt_vocoder = Path(ckpt_vocoder)
        vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')

        with open(ckpt_vocoder / 'args.yml', 'r') as f:
            vocoder_args = yaml.load(f, Loader=yaml.UnsafeLoader)

        self.generator = Generator(vocoder_args.n_mel_channels, vocoder_args.ngf,
                                   vocoder_args.n_residual_layers)
        self.generator.load_state_dict(vocoder_sd)
        self.generator.eval()

    def vocode(self, spec, global_step=None):
        with torch.no_grad():
            return self.generator(torch.from_numpy(spec).unsqueeze(0)).squeeze().numpy()

class VocoderGriffinLim(object):

    def __init__(self, spec_dir_name):
        self.spec_dir_name = spec_dir_name

    def vocode(self, spec, global_step):
        # inv_transform may stuck when the mel spec is bad. We time it out and replace with other sound
        signal.signal(signal.SIGALRM, self.timeout_handler)
        # no need to wait long time during the first couple of epochs
        if global_step < 4096:
            signal.alarm(7)
        else:
            signal.alarm(30)
        try:
            wave = inv_transforms(spec, self.spec_dir_name)
            signal.alarm(0)
        except TimeoutError as msg:
            wave, _ = librosa.load('./data/10s_rick_roll_22050.wav', sr=None)
            print(msg)
        return wave

    @classmethod
    def timeout_handler(signum, frame):
        raise TimeoutError('Bad spec: took too much time to reconstruct the sound from spectrogram')


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 for_specs=False, vocoder_cfg=None, spec_dir_name=None, sample_rate=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.for_specs = for_specs
        self.spec_dir_name = spec_dir_name
        self.sample_rate = sample_rate
        print('We will not save audio for conditioning and conditioning_rec')
        if self.for_specs:
            self.vocoder = instantiate_from_config(vocoder_cfg)

    def _visualize_attention(self, attention, scale_by_prior=True):
        if scale_by_prior:
            B, H, T, T = attention.shape
            # attention weight is 1/T: if we have a seq with length 3 the weights are 1/3, 1/3, and 1/3
            # making T by T matrix with zeros in the upper triangular part
            attention_uniform_prior = 1 / torch.arange(1, T+1).view(1, T, 1).repeat(B, 1, T)
            attention_uniform_prior = attention_uniform_prior.tril().view(B, 1, T, T).to(attention.device)
            attention = attention - attention_uniform_prior

        attention_agg = attention.sum(dim=1, keepdims=True)
        return attention_agg

    def _log_rec_audio(self, specs, tag, global_step, pl_module=None, save_rec_path=None):

        # specs are (B, 1, F, T)
        for i, spec in enumerate(specs):
            spec = spec.data.squeeze(0).cpu().numpy()
            # audios are in [-1, 1], making them in [0, 1]
            spec = (spec + 1) / 2
            wave = self.vocoder.vocode(spec, global_step)
            wave = torch.from_numpy(wave).unsqueeze(0)
            if pl_module is not None:
                pl_module.logger.experiment.add_audio(f'{tag}_{i}', wave, pl_module.global_step, self.sample_rate)
            # in case we would like to save it on disk
            if save_rec_path is not None:
                try:
                    librosa.output.write_wav(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate)
                except AttributeError:
                    soundfile.write(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate, 'FLOAT')

    @rank_zero_only
    def _testtube(self, pl_module, images, batch, batch_idx, split):

        if pl_module.__class__.__name__ == 'Net2NetTransformer':
            cond_stage_model = pl_module.cond_stage_model.__class__.__name__
        else:
            cond_stage_model = None

        for k in images:
            tag = f'{split}/{k}'
            if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
                # saving the classes for the current batch
                pl_module.logger.experiment.add_text(tag, '; '.join(batch['label']))
                # breaking here because we don't want to call add_image
                if cond_stage_model == 'FeatsClassStage':
                    grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
                else:
                    continue
            elif k in ['att_nopix', 'att_half', 'att_det']:
                B, H, T, T = images[k].shape
                grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
            elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
                grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
            else:
                if self.for_specs:
                    # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
                    grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
                    # also reconstruct waveform given the spec and inv_transform
                    if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                        self._log_rec_audio(images[k], tag, pl_module.global_step, pl_module=pl_module)
                else:
                    grid = torchvision.utils.make_grid(images[k])
                # attention is already in [0, 1] therefore ignoring this line
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, pl_module, split, images, batch, batch_idx):
        root = os.path.join(pl_module.logger.save_dir, 'images', split)

        if pl_module.__class__.__name__ == 'Net2NetTransformer':
            cond_stage_model = pl_module.cond_stage_model.__class__.__name__
        else:
            cond_stage_model = None

        for k in images:
            if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
                filename = '{}_gs-{:06}_e-{:03}_b-{:06}.txt'.format(
                    k,
                    pl_module.global_step,
                    pl_module.current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                # saving the classes for the current batch
                with open(path, 'w') as file:
                    file.write('\n'.join(batch['label']))
                # next loop iteration here because we don't want to call add_image
                if cond_stage_model == 'FeatsClassStage':
                    grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
                else:
                    continue
            elif k in ['att_nopix', 'att_half', 'att_det']:  # GPT CLass
                B, H, T, T = images[k].shape
                grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
            elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
                grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
            else:
                if self.for_specs:
                    # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
                    grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
                else:
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                # attention is already in [0, 1] therefore ignoring this line
                grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:03}_b-{:06}.png'.format(
                k,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

            # also save audio on the disk
            if self.for_specs:
                tag = f'{split}/{k}'
                filename = filename.replace('.png', '.wav')
                path = os.path.join(root, filename)
                if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                    self._log_rec_audio(images[k], tag, pl_module.global_step, save_rec_path=path)

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and
                self.max_images > 0 and
                pl_module.first_stage_key != 'feature'):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                if isinstance(images[k], dict):
                    N = min(images[k]['feature'].shape[0], self.max_images)
                    images[k]['feature'] = images[k]['feature'][:N]
                    if isinstance(images[k]['feature'], torch.Tensor):
                        images[k]['feature'] = images[k]['feature'].detach().cpu()
                        if self.clamp:
                            images[k]['feature'] = torch.clamp(images[k]['feature'], -1., 1.)
                else:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module, split, images, batch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, batch, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split='val')


if __name__ == '__main__':
    # adding a random number of seconds so that exp folder names coincide less often
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')

    # add cwd for convenience and to make classes in this file available when
    # running as `python train.py`
    # (in particular `train.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            '-n/--name and -r/--resume cannot be specified both.'
            'If you want to resume training in a new log folder, '
            'use -n/--name in combination with --resume_from_checkpoint'
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError('Cannot find {}'.format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split('/')
            idx = len(paths)-paths[::-1].index('logs')+1
            logdir = '/'.join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip('/')
            # ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')
            # ckpt = sorted(glob.glob(os.path.join(logdir, 'checkpoints', '*.ckpt')))[-1]
            if Path(os.path.join(logdir, 'checkpoints', 'last.ckpt')).exists():
                ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')
            else:
                ckpt = sorted(Path(logdir).glob('checkpoints/*.ckpt'))[-1]

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, 'configs/*.yaml')))
        opt.base = base_configs+opt.base
        _tmp = logdir.split('/')
        nowname = _tmp[_tmp.index('logs')+1]
    else:
        if opt.name:
            name = '_'+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = '_'+cfg_name
        else:
            name = ''
        nowname = now+name+opt.postfix
        logdir = os.path.join('logs', nowname)

    print(nowname)
    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop('lightning', OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get('trainer', OmegaConf.create())
        # default to ddp
        trainer_config['distributed_backend'] = 'ddp'
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if 'gpus' not in trainer_config:
            del trainer_config['distributed_backend']
            cpu = True
        else:
            gpuinfo = trainer_config['gpus']
            print(f'Running on GPUs {gpuinfo}')
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            'testtube': {
                'target': 'pytorch_lightning.loggers.TestTubeLogger',
                'params': {
                    'name': 'testtube',
                    'save_dir': logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs['testtube']
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            'target': 'pytorch_lightning.callbacks.ModelCheckpoint',
            'params': {
                'dirpath': ckptdir,
                'filename': '{epoch:06}',
                'verbose': True,
                'save_last': True,
            }
        }
        if hasattr(model, 'monitor'):
            print(f'Monitoring {model.monitor} as checkpoint metric.')
            default_modelckpt_cfg['params']['monitor'] = model.monitor
            default_modelckpt_cfg['params']['save_top_k'] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs['checkpoint_callback'] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            'setup_callback': {
                'target': 'train.SetupCallback',
                'params': {
                    'resume': opt.resume,
                    'now': now,
                    'logdir': logdir,
                    'ckptdir': ckptdir,
                    'cfgdir': cfgdir,
                    'config': config,
                    'lightning_config': lightning_config,
                }
            },
            'image_logger': {
                'target': 'train.ImageLogger',
                'params': {
                    'batch_frequency': 750,
                    'max_images': 4,
                    'clamp': True
                }
            },
            'learning_rate_logger': {
                'target': 'train.LearningRateMonitor',
                'params': {
                    'logging_interval': 'step',
                    #'log_momentum': True
                }
            },
        }
        # patching the default config for the spectrogram input
        if 'Spectrogram' in config.data.target:
            spec_dir_name = Path(config.data.params.spec_dir_path).name
            default_callbacks_cfg['image_logger']['params']['spec_dir_name'] = spec_dir_name
            default_callbacks_cfg['image_logger']['params']['sample_rate'] = config.data.params.sample_rate

        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs['callbacks'] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(',').split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f'accumulate_grad_batches = {accumulate_grad_batches}')
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print('Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'.format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print('Summoning checkpoint.')
                ckpt_path = os.path.join(ckptdir, 'last.ckpt')
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, 'debug_runs', name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
