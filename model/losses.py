from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class CustomLoss(nn.Module, ABC):
    def forward(self, x_real, y_real, generator, discriminator):
        pass


class Pix2PixHDLoss(CustomLoss):
    def __init__(self, device, lambda1=10., lambda2=10., norm_weight_to_one=True):
        super().__init__()

        lambda0 = 1.0
        # Keep ratio of composite loss, but scale down max to 1.0
        scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0
        self.device = device

        self.lambda0 = lambda0 / scale
        self.lambda1 = lambda1 / scale
        self.lambda2 = lambda2 / scale

    def adv_loss(self, discriminator_preds, is_real):
        '''
        Computes adversarial loss from nested list of fakes outputs from discriminator.
        '''
        target = torch.ones_like if is_real else torch.zeros_like

        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += F.mse_loss(pred, target(pred))
        return adv_loss

    def fm_loss(self, real_preds, fake_preds):
        '''
        Computes feature matching loss from nested lists of fake and real outputs from discriminator.
        '''

        fm_loss = 0.0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += F.l1_loss(real_feature.detach(), fake_feature)
        return fm_loss

    def forward(self, x_real, y_real, generator, discriminator):
        '''
        Function that computes the forward pass and total loss for generator and discriminator.
        '''

        with autocast():
            y_fake = generator(x_real)

            # Get necessary outputs for loss/backprop for both generator and discriminator
            d_on_real = discriminator(y_real)
            d_on_fake = discriminator(y_fake)

        g_loss = (
                self.lambda0 * self.adv_loss(d_on_fake, True) +
                self.lambda1 * self.fm_loss(d_on_real, d_on_fake) / discriminator.module.n_discriminators
        )
        d_loss = 0.5 * (
                self.adv_loss(d_on_real, True) +
                self.adv_loss(d_on_fake, False)
        )

        return g_loss, d_loss, y_fake.detach()


class Pix2PixLoss(CustomLoss):
    def __init__(self, device, reconstruction_loss_weight):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.device = device
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, x_real, y_real, generator, discriminator, logger=None):
        batch_size = y_real.shape[0]

        real_label = torch.ones(batch_size, 1, 3, 31, 31).float().to(self.device)
        fake_label = torch.zeros(batch_size, 1, 3, 31, 31).float().to(self.device)

        # Use mixed precision
        with autocast():
            with torch.no_grad():
                fake = generator(x_real)
            disc_fake_hat = discriminator(fake.detach(), x_real)
            disc_real_hat = discriminator(y_real, x_real)

        disc_fake_loss = self.adv_criterion(disc_fake_hat, fake_label)
        disc_real_loss = self.adv_criterion(disc_real_hat, real_label)
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        with autocast():
            fake = generator(x_real)
            gen_adv_hat = discriminator(fake, x_real)

        gen_adv_loss = self.adv_criterion(gen_adv_hat, real_label)
        gen_recon_loss = self.recon_criterion(y_real, fake)
        gen_loss = gen_adv_loss + gen_recon_loss * self.reconstruction_loss_weight

        return gen_loss, disc_loss, fake.detach()
