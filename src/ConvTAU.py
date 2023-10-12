
import math
import torch
from torch import optim, nn
import lightning.pytorch as pl

from src.parameters import params_ConvTAU
from src.OpticalFlow import OpticalFlowEstimator, OpticalLayer

# Code customized and extended from: https://github.com/chengtan9907/OpenSTL


class Conv2d_layer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 dilation=1):
        super(Conv2d_layer, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.act(self.norm(y))
        return y


class Encoder(nn.Module):

    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super(Encoder, self).__init__()
        samplings = [False, True] * (N_S // 2)
        self.enc = nn.Sequential(
            Conv2d_layer(C_in, C_hid, spatio_kernel,
                         downsampling=samplings[0]),
            *[Conv2d_layer(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):

    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super(Decoder, self).__init__()
        samplings = list(reversed([False, True] * (N_S // 2)))
        self.dec = nn.Sequential(
            *[Conv2d_layer(C_hid, C_hid, spatio_kernel, upsampling=s)
              for s in samplings[:-1]],
            Conv2d_layer(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        y = self.dec[-1](hid + enc1)
        y = self.readout(y)
        return y


class TemporalAttentionModule(nn.Module):

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)  # 1x1 conv
        self.activation = nn.ReLU()  # nn.GELU()

        # Statical Attention Layers
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = int(kernel_size // dilation +
                   ((kernel_size // dilation) % 2 - 1))
        dd_p = (dilation * (dd_k - 1) // 2)
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p,
                               groups=dim)  # depth-wise conv
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)  # depth-wise dilation conv
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1 conv

        # Dynamical Attention Layers
        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid()
        )

        self.proj_2 = nn.Conv2d(dim, dim, 1)  # 1x1 conv

    def forward(self, x):
        skip1 = x.clone()

        x = self.activation(self.proj_1(x))
        skip2 = x.clone()

        # Statical Attention
        statical_attn = self.conv0(x)
        statical_attn = self.conv_spatial(statical_attn)
        statical_attn = self.conv1(statical_attn)

        # Dynamical Attention
        b, c, _, _ = x.size()
        dynamical_attn = self.avg_pool(x).view(b, c)
        dynamical_attn = self.fc(dynamical_attn).view(b, c, 1, 1)

        attn = dynamical_attn * statical_attn * skip2

        return self.proj_2(attn) + skip1


class MixMlp(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  # 1x1

    def forward(self, x):
        x = self.fc2(self.act(self.dwconv(self.fc1(x))))
        return x


class TAUModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=21, mlp_ratio=8., init_value=1e-2):
        super(TAUModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.attn = TemporalAttentionModule(in_channels, kernel_size)

        self.norm2 = nn.BatchNorm2d(in_channels)
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = MixMlp(in_features=in_channels,
                          hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((in_channels)), requires_grad=True)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = x + \
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * \
            self.attn(self.norm1(x))
        z = z + \
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * \
            self.mlp(self.norm2(z))
        return z if self.in_channels == self.out_channels else self.reduction(z)


class TAU(nn.Module):
    def __init__(self, channel_in, channel_hid, N_TAUModules, mlp_ratio):
        super(TAU, self).__init__()
        assert N_TAUModules >= 2 and mlp_ratio > 1
        self.N2 = N_TAUModules

        # downsample
        enc_layers = [TAUModule(channel_in, channel_hid, mlp_ratio)]
        # middle layers
        for i in range(1, N_TAUModules-1):
            enc_layers.append(TAUModule(channel_hid, channel_hid, mlp_ratio))
        # upsample
        enc_layers.append(TAUModule(channel_hid, channel_in, mlp_ratio))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class ConvTAU(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.mse = nn.MSELoss()  # to focus on intra-frame-level differences
        T, C, H, W = params['in_shape']
        # downsample 1 / 2**(N_S/2)
        H, W = int(H / 2**(params['N_S']/2)), int(W / 2**(params['N_S']/2))

        self.enc = Encoder(
            C + 2, params['hid_S'], params['N_S'], params['spatio_kernel_enc'])
        self.hid = TAU(T*params['hid_S'], params['hid_T'], params['N_T'],
                       mlp_ratio=params['mlp_ratio'])
        self.dec = Decoder(params['hid_S'], C,
                           params['N_S'], params['spatio_kernel_dec'])

        self.flow_estimator = OpticalLayer()

    def forward(self, x):
        x = x.unsqueeze(2)
        B, T, C, H, W = x.shape

        # Optical Flow
        flow = self.flow_estimator(x)

        # Encoding
        x = x.view(B*T, C, H, W)
        x = torch.cat((x, flow), 1)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # TAU
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        # Decoding
        out = self.dec(hid, skip)
        out = out.reshape(B, T, C, H, W)

        return out

    def single_prediction(self, frames):
        frames = frames.unsqueeze(0).float()
        out = self(frames)
        return out.squeeze(0).squeeze(1).detach()

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        # x = x / 255.0
        # y = y / 255.0
        out = self(x)

        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mse_loss", mse_loss, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        # x = x / 255.0
        # y = y / 255.0
        out = self(x)

        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_mse_loss", mse_loss, on_epoch=True)
        self.log("validation_kl_loss", kl_loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        # x = x / 255.0
        # y = y / 255.0
        out = self(x)
        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_mse_loss", mse_loss, on_epoch=True)
        self.log("test_kl_loss", kl_loss, on_epoch=True)
        # out = out * 255.0
        # y = y * 255.0
        # loss_not_normalized, _, _ = self.loss(out, y)
        # self.log("test_loss_notNormalized", loss_not_normalized, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(
        ), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
        return optimizer

    # to focus on inter-frame-level differences
    def kullback_leibler_divergence(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T = pred_y.shape[:2]
        if T <= 2:
            return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = nn.functional.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = nn.functional.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()

    def loss(self, pred, y):
        mse_loss = self.mse(pred, y)
        kl_loss = self.params['kl_divergence_weight'] * \
            self.kullback_leibler_divergence(pred, y)
        loss = mse_loss + kl_loss
        return (loss, mse_loss, kl_loss)
