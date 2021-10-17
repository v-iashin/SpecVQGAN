import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import sys
sys.path.insert(0, '.')  # nopep8
from train import instantiate_from_config

from specvqgan.modules.diffusionmodules.model import Encoder, Decoder, Encoder1d, Decoder1d
from specvqgan.modules.vqvae.quantize import VectorQuantizer, VectorQuantizer1d


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super().__init__()
        self.image_key = image_key
        # we need this one for compatibility in train.ImageLogger.log_img if statement
        self.first_stage_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.used_codes = []
        self.counts = [0 for _ in range(self.quantize.n_e)]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)  # 2d: (B, 256, 16, 16) <- (B, 3, 256, 256)
        h = self.quant_conv(h)  # 2d: (B, 256, 16, 16)
        quant, emb_loss, info = self.quantize(h)  # (B, 256, 16, 16), (), ((), (768, 1024), (768, 1))
        if not self.training:
            self.counts = [info[2].squeeze().tolist().count(i) + self.counts[i] for i in range(self.quantize.n_e)]
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.global_step != 0 and sum(self.counts) > 0:
            print(f'Previous Epoch counts: {self.counts}')
            zero_hit_codes = len([1 for count in self.counts if count == 0])
            used_codes = []
            for c, count in enumerate(self.counts):
                used_codes.extend([c] * count)
            self.logger.experiment.add_histogram('val/code_hits', torch.tensor(used_codes), self.global_step)
            self.logger.experiment.add_scalar('val/zero_hit_codes', zero_hit_codes, self.global_step)
            self.counts = [0 for _ in range(self.quantize.n_e)]
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae['val/rec_loss']
        self.log('val/rec_loss', rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val/aeloss', aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModel1d(VQModel):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[],
                 image_key='feature', colorize_nlabels=None, monitor=None):
        # ckpt_path is none to super because otherwise will try to load 1D checkpoint into 2D model
        super().__init__(ddconfig, lossconfig, n_embed, embed_dim)
        self.image_key = image_key
        # we need this one for compatibility in train.ImageLogger.log_img if statement
        self.first_stage_key = image_key
        self.encoder = Encoder1d(**ddconfig)
        self.decoder = Decoder1d(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer1d(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv1d(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig['z_channels'], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def get_input(self, batch, k):
        x = batch[k]
        if self.image_key == 'feature':
            x = x.permute(0, 2, 1)
        elif self.image_key == 'image':
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def forward(self, input):
        if self.image_key == 'image':
            input = input.squeeze(1)
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        if self.image_key == 'image':
            dec = dec.unsqueeze(1)
        return dec, diff

    def log_images(self, batch, **kwargs):
        if self.image_key == 'image':
            log = dict()
            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec, _ = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log['inputs'] = x
            log['reconstructions'] = xrec
            return log
        else:
            raise NotImplementedError('1d input should be treated differently')

    def to_rgb(self, batch, **kwargs):
        raise NotImplementedError('1d input should be treated differently')


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from train import instantiate_from_config

    image_key = 'image'
    cfg_audio = OmegaConf.load('./configs/vggsound_codebook.yaml')
    model = VQModel(cfg_audio.model.params.ddconfig,
                    cfg_audio.model.params.lossconfig,
                    cfg_audio.model.params.n_embed,
                    cfg_audio.model.params.embed_dim,
                    image_key='image')
    batch = {
        'image': torch.rand((4, 80, 848)),
        'file_path_': ['data/vggsound/mel123.npy', 'data/vggsound/mel123.npy', 'data/vggsound/mel123.npy'],
        'class': [1, 1, 1],
    }
    xrec, qloss = model(model.get_input(batch, image_key))
    print(xrec.shape, qloss.shape)
