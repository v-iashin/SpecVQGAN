import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.discriminator.model import (NLayerDiscriminator, NLayerDiscriminator1dFeats,
                                                   NLayerDiscriminator1dSpecs,
                                                   weights_init)
from specvqgan.modules.losses.lpaps import LPAPS


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPAPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", min_adapt_weight=0.0, max_adapt_weight=1e4):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPAPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPAPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.min_adapt_weight = min_adapt_weight
        self.max_adapt_weight = max_adapt_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, self.min_adapt_weight, self.max_adapt_weight).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPAPSWithDiscriminator1dFeats(VQLPAPSWithDiscriminator):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", min_adapt_weight=0.0, max_adapt_weight=1e4):
        super().__init__(disc_start=disc_start, codebook_weight=codebook_weight,
                         pixelloss_weight=pixelloss_weight, disc_num_layers=disc_num_layers,
                         disc_in_channels=disc_in_channels, disc_factor=disc_factor, disc_weight=disc_weight,
                         perceptual_weight=perceptual_weight, use_actnorm=use_actnorm,
                         disc_conditional=disc_conditional, disc_ndf=disc_ndf, disc_loss=disc_loss,
                         min_adapt_weight=min_adapt_weight, max_adapt_weight=max_adapt_weight)

        self.discriminator = NLayerDiscriminator1dFeats(input_nc=disc_in_channels, n_layers=disc_num_layers,
                                                   use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)

class VQLPAPSWithDiscriminator1dSpecs(VQLPAPSWithDiscriminator):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", min_adapt_weight=0.0, max_adapt_weight=1e4):
        super().__init__(disc_start=disc_start, codebook_weight=codebook_weight,
                         pixelloss_weight=pixelloss_weight, disc_num_layers=disc_num_layers,
                         disc_in_channels=disc_in_channels, disc_factor=disc_factor, disc_weight=disc_weight,
                         perceptual_weight=perceptual_weight, use_actnorm=use_actnorm,
                         disc_conditional=disc_conditional, disc_ndf=disc_ndf, disc_loss=disc_loss,
                         min_adapt_weight=min_adapt_weight, max_adapt_weight=max_adapt_weight)

        self.discriminator = NLayerDiscriminator1dSpecs(input_nc=disc_in_channels, n_layers=disc_num_layers,
                                                   use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)


if __name__ == '__main__':
    from specvqgan.modules.diffusionmodules.model import Decoder, Decoder1d

    optimizer_idx = 0
    loss_config = {
        'disc_conditional': False,
        'disc_start': 30001,
        'disc_weight': 0.8,
        'codebook_weight': 1.0,
    }
    ddconfig = {
        'ch': 128,
        'num_res_blocks': 2,
        'dropout': 0.0,
        'z_channels': 256,
        'double_z': False,
    }
    qloss = torch.rand(1, requires_grad=True)

    ## AUDIO
    loss_config['disc_in_channels'] = 1
    ddconfig['in_channels'] = 1
    ddconfig['resolution'] = 848
    ddconfig['attn_resolutions'] = [53]
    ddconfig['out_ch'] = 1
    ddconfig['ch_mult'] = [1, 1, 2, 2, 4]
    decoder = Decoder(**ddconfig)
    loss = VQLPAPSWithDiscriminator(**loss_config)
    x = torch.rand(16, 1, 80, 848)
    # subtracting something which uses dec_conv_out so that it will be in a graph
    xrec = torch.rand(16, 1, 80, 848) - decoder.conv_out(torch.rand(16, 128, 80, 848)).mean()
    aeloss, log_dict_ae = loss(qloss, x, xrec, optimizer_idx, global_step=0,last_layer=decoder.conv_out.weight)
    print(aeloss)
    print(log_dict_ae)
