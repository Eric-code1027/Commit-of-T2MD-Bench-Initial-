# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.
#
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
#
# NOTE: This file is vendored into DiffSynth-Studio to avoid external repo dependency.
# It provides BigVGANFlowVAE + init_vae_stat(checkpoint_path, stat_path, device).

import json
import math
from typing import Optional, Union

import torch
from torch import pow, sin
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from .alias_free_activation.torch.act import Activation1d as TorchActivation1d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class Conv1d_S(nn.Module):
    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        norm_type="weight_norm",
        init_type=None,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)

        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = spectral_norm(self.layer)

    def forward(self, inputs):
        return self.layer(inputs)


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=base**i,
                            padding=base**i,
                        )
                    ),
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=1,
                            padding=1,
                        )
                    ),
                )
                for i in range(nums)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=(12, 24, 48, 96, 192, 384, 768),
        down_sample_factors=(2, 2, 2, 2, 4, 4),
    ):
        super().__init__()
        act_slope = 0.2
        layers = []
        layers += [
            Conv1d_S(in_channels, base_channels, kernel_size=proj_kernel_size, stride=1),
            nn.LeakyReLU(act_slope, True),
        ]

        for (in_c, out_c), down_f in zip(zip(channels[:-1], channels[1:]), down_sample_factors):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(act_slope, True),
            ]

        layers += [
            Conv1d_S(channels[-1], out_channels, proj_kernel_size, stride=1),
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, z_inputs=None):
        return self.generator(conditions)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        bn: bool = False,
        activation=None,
        w_init_gain=None,
        input_transpose: bool = False,
        **kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        outputs = self.activation(self.bn(super().forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose
        self.in_channels = in_channels

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super().forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x.transpose(1, 2) if self.transpose else x


class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        h,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        causal: bool = True,
        act_causal: bool = False,
    ):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, stride=1, dilation=d, causal=causal)
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, stride=1, dilation=1, causal=causal)
                )
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)

        Activation1d = TorchActivation1d
        if activation == "snake":
            self.activations = nn.ModuleList(
                [Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale), causal=act_causal) for _ in range(self.num_layers)]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale), causal=act_causal) for _ in range(self.num_layers)]
            )
        else:
            raise NotImplementedError("activation incorrectly specified")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(
        self,
        h,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        causal: bool = True,
        act_causal: bool = False,
    ):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=d, causal=causal)) for d in dilation]
        )
        self.num_layers = len(self.convs)

        Activation1d = TorchActivation1d
        if activation == "snake":
            self.activations = nn.ModuleList(
                [Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale), causal=act_causal) for _ in range(self.num_layers)]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale), causal=act_causal) for _ in range(self.num_layers)]
            )
        else:
            raise NotImplementedError("activation incorrectly specified")

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGANFlowVAE(nn.Module):
    def __init__(self, h, stat_path=None, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel
        causal = h.causal
        act_causal = h.get("act_causal", False)

        self.normalize_latent = False
        if stat_path:
            self.normalize_latent = True
            data = torch.load(stat_path)
            self.register_buffer("latent_mean", data["mean"].float().view(1, -1, 1))
            self.register_buffer("latent_std", data["var"].float().sqrt().view(1, -1, 1))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim * 2,
            channels=h.downsample_channels,
            down_sample_factors=h.downsample_rates,
        )

        Activation1d = TorchActivation1d
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False))

        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock {h.resblock}")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation, causal=causal, act_causal=act_causal)
                )

        activation_post = (
            Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snakebeta"
            else None
        )
        if activation_post is None:
            raise NotImplementedError("activation incorrectly specified")
        self.activation_post = Activation1d(activation=activation_post, causal=act_causal)

        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal, bias=self.use_bias_at_final))

        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    @torch.no_grad()
    def encode(self, x):
        x = self.audio_encoder(x)
        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        if self.normalize_latent:
            z = (z - self.latent_mean) / self.latent_std
        return z

    @torch.no_grad()
    def decode(self, z):
        if self.normalize_latent:
            z = z * self.latent_std + self.latent_mean

        x = self.conv_pre(z)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def init_vae_stat(checkpoint_path: str, stat_path: Optional[str] = None, device: Union[int, str] = 0):
    config = """{
        "resblock": "1",
        "upsample_rates": [5,4,3,2,2,2],
        "upsample_kernel_sizes": [10,8,6,4,4,4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "downsample_rates": [2,2,2,3,4,5],
        "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
        "use_tanh_at_final": false,
        "use_bias_at_final": false,
        "activation": "snakebeta",
        "snake_logscale": true,
        "causal": true,
        "act_causal": true,
        "latent_dim": 64,
        "sampling_rate": 24000
    }"""

    h = AttrDict(json.loads(config))
    torch.backends.cudnn.benchmark = False

    generator = BigVGANFlowVAE(h, stat_path)
    state_dict_g = torch.load(checkpoint_path, map_location="cpu")
    sd = state_dict_g["generator"] if isinstance(state_dict_g, dict) and "generator" in state_dict_g else state_dict_g
    missing, unexpected = generator.load_state_dict(sd, strict=False)
    for name in missing:
        assert name.startswith("latent")
    for name in unexpected:
        assert name.startswith("flow")
    print(f"BigVGAN VAE Missing parameters: {missing}")
    generator.remove_weight_norm()
    generator.requires_grad_(False).eval()
    return generator.to(device)


