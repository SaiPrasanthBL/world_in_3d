from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    functional as F,
    Conv2d,
    LeakyReLU,
    Upsample,
    Sigmoid,
    ConvTranspose2d,
)

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(
        self,
        pad_reflection=True,
        gaussian_average=False,
        comp_mode=False,
        eval_mode=False,
    ):
        super(SSIM, self).__init__()
        self.comp_mode = comp_mode
        self.eval_mode = eval_mode

        if not gaussian_average:
            self.mu_x_pool = nn.AvgPool2d(3, 1)
            self.mu_y_pool = nn.AvgPool2d(3, 1)
            self.sig_x_pool = nn.AvgPool2d(3, 1)
            self.sig_y_pool = nn.AvgPool2d(3, 1)
            self.sig_xy_pool = nn.AvgPool2d(3, 1)
        else:
            self.mu_x_pool = GaussianAverage()
            self.mu_y_pool = GaussianAverage()
            self.sig_x_pool = GaussianAverage()
            self.sig_y_pool = GaussianAverage()
            self.sig_xy_pool = GaussianAverage()

        if pad_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y, pad=True):
        if pad:
            x = self.pad(x)
            y = self.pad(y)
        ## average of pixels in x and y, average pooling or Gaussian averaging, based on the initialization
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        mu_x_sq = (
            mu_x**2
        )  ## squares of the averages and the product of the averages, respectively.
        mu_y_sq = mu_y**2
        mu_x_y = mu_x * mu_y
        ## variances and covariance:
        sigma_x = self.sig_x_pool(x**2) - mu_x_sq
        sigma_y = self.sig_y_pool(y**2) - mu_y_sq
        sigma_xy = self.sig_xy_pool(x * y) - mu_x_y

        SSIM_n = (2 * mu_x_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)

        if (
            not self.eval_mode
        ):  ## determines how to handle the output of the SSIM calculation
            if (
                not self.comp_mode
            ):  ## error (1 - SSIM index), used as a loss function during training
                return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
            else:  ## return the SSIM index itself.
                return torch.clamp((1 - SSIM_n / SSIM_d), 0, 1) / 2
        else:
            return (
                SSIM_n / SSIM_d
            )  ## returned error is scaled to range between 0 and 1 for easier interpretation and usage in loss calculations
        ## (2*mu_x*mu_y + C1)*(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x + sigma_y + C2)).
        ## SSIM index ranges from -1 to 1, where 1 means the images are identical, -1 means the images are totally different, and 0 means the images are not correlated


class GEO(nn.Module):
    """Layer to compute the pseudo label, L_{geo}, loss between a pair of images"""

    def __init__(
        self,
        pad_reflection=True,
        gaussian_average=False,
        comp_mode=False,
        eval_mode=False,
    ):
        super(GEO, self).__init__()
        self.comp_mode = comp_mode
        self.eval_mode = eval_mode

        if not gaussian_average:
            self.mu_x_pool = nn.AvgPool2d(3, 1)
            self.mu_y_pool = nn.AvgPool2d(3, 1)
            self.sig_x_pool = nn.AvgPool2d(3, 1)
            self.sig_y_pool = nn.AvgPool2d(3, 1)
            self.sig_xy_pool = nn.AvgPool2d(3, 1)
        else:
            self.mu_x_pool = GaussianAverage()
            self.mu_y_pool = GaussianAverage()
            self.sig_x_pool = GaussianAverage()
            self.sig_y_pool = GaussianAverage()
            self.sig_xy_pool = GaussianAverage()

        if pad_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y, pad=True):
        if pad:
            x = self.pad(x)
            y = self.pad(y)
        ## average of pixels in x and y, average pooling or Gaussian averaging, based on the initialization
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        mu_x_sq = (
            mu_x**2
        )  ## squares of the averages and the product of the averages, respectively.
        mu_y_sq = mu_y**2
        mu_x_y = mu_x * mu_y
        ## variances and covariance:
        sigma_x = self.sig_x_pool(x**2) - mu_x_sq
        sigma_y = self.sig_y_pool(y**2) - mu_y_sq
        sigma_xy = self.sig_xy_pool(x * y) - mu_x_y

        SSIM_n = (2 * mu_x_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)

        if (
            not self.eval_mode
        ):  ## determines how to handle the output of the SSIM calculation
            if (
                not self.comp_mode
            ):  ## error (1 - SSIM index), used as a loss function during training
                return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
            else:  ## return the SSIM index itself.
                return torch.clamp((1 - SSIM_n / SSIM_d), 0, 1) / 2
        else:
            return (
                SSIM_n / SSIM_d
            )  ## returned error is scaled to range between 0 and 1 for easier interpretation and usage in loss calculations
        ## (2*mu_x*mu_y + C1)*(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x + sigma_y + C2)).
        ## SSIM index ranges from -1 to 1, where 1 means the images are identical, -1 means the images are totally different, and 0 means the images are not correlated


def ssim(
    x,
    y,
    pad_reflection=True,
    gaussian_average=False,
    comp_mode=False,
    eval_mode=False,
    pad=True,
):
    ssim_ = SSIM(pad_reflection, gaussian_average, comp_mode, eval_mode)
    return ssim_(x, y, pad=pad)


def geo(
    x,
    y,
    pad_reflection=True,
    gaussian_average=False,
    comp_mode=False,
    eval_mode=False,
    pad=True,
):
    geo_ = GEO(pad_reflection, gaussian_average, comp_mode, eval_mode)
    return geo_(x, y, pad=pad)