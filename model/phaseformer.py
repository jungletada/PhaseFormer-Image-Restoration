import torch
import numpy as np
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from model.spade_decoder import SpadeDecoder


def _cfg(url='', **kwargs):
    return {
        'crop_pct': .96, 
        **kwargs
    }


default_cfgs = {
    'phaseformer_T': _cfg(crop_pct=0.9),
    'phaseformer_S': _cfg(crop_pct=0.9),
    'phaseformer_M': _cfg(crop_pct=0.9),
    'phaseformer_B': _cfg(crop_pct=0.875),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=False):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, bias=bias)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LargeKernelConv(nn.Module):
    def __init__(self, dim, kernel_size=9, dilation=2, bias=False):
        super(LargeKernelConv, self).__init__()
        kernel_dw = int(2 * dilation - 1)
        padding_dw = kernel_dw //2
        kernel_dw_d = int(np.ceil(kernel_size / dilation))
        padding_dw_d = ((kernel_dw_d  - 1) * dilation + 1) // 2
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_dw,
                                 stride=1, padding=padding_dw, groups=dim, bias=bias)
        self.dw_d_conv = nn.Conv2d(dim, dim, kernel_size=kernel_dw_d,
                                   stride=1, padding=padding_dw_d, dilation=dilation, groups=dim, bias=bias)
        self.conv_fc = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_d_conv(x)
        x = self.conv_fc(x)
        return x


class CosAttention(nn.Module):
    def __init__(self, dim, kernel_size, dilation, proj_drop=0., bias=False):
        super().__init__()
        self.lk_amplitude = nn.Sequential(
            LargeKernelConv(dim=dim, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm2d(dim),
            nn.GELU())
        self.lk_theta = LargeKernelConv(dim=dim, kernel_size=kernel_size, dilation=dilation)
        self.fc_amplitude = nn.Conv2d(dim, dim, 1, 1, bias=bias)
        self.fc_theta = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.proj_drop = nn.Dropout2d(proj_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=bias)
        self.reweight = nn.Conv2d(4*dim, dim, 1, 1, bias=bias)

    def forward(self, x):
        aplt1 = self.lk_amplitude(x)
        theta1 = self.lk_theta(x)
        aplt2 = self.fc_amplitude(x)
        theta2 = self.fc_theta(x)
        s = torch.cat((aplt1*torch.cos(theta1),
                       aplt1 * torch.sin(theta1),
                       aplt2*torch.cos(theta2),
                       aplt2 * torch.sin(theta2),
                       ), dim=1)
        x = self.reweight(s) + x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CosAttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., bias=False, kernel_size=9, dilation=2,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CosAttention(dim, kernel_size=kernel_size, dilation=dilation, bias=bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlappingPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1, in_dims=3, embed_dim=64,
                 norm_layer=nn.BatchNorm2d, groups=1, use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_dims, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=groups),
        )
        self.norm = norm_layer(embed_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_layer(out_embed_dim) if use_norm == True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=4., bias=False, kernel_size=9, dilation=2,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, ):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(CosAttentionBlock(dim, mlp_ratio=mlp_ratio, kernel_size=kernel_size, dilation=dilation,
                                        bias=bias, drop_path=block_dpr, norm_layer=norm_layer))
    blocks = nn.Sequential(*blocks)
    return blocks


class PhaseFormer(nn.Module):
    def __init__(self, layers,
                 embed_dims=None, mlp_ratios=None,
                 bias=False, drop_path_rate=0., in_chans=3,
                 kernel_sizes=(11, 9, 7, 5), dilations=(2, 2, 2, 2),
                 norm_layer=nn.BatchNorm2d, ds_use_norm=True, downsample=4):
        super().__init__()
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.layers = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()
        self.num_stages = len(layers)

        if downsample == 4:
            stride = 2
        elif downsample == 3:
            stride = 1
        else:
            print("downsample rate shoule be 3 or 4.")
            raise NotImplementedError

        for i in range(self.num_stages):
            if i == 0:
                self.patch_embeds.append(OverlappingPatchEmbed(
                    patch_size=3, stride=stride, padding=1, in_dims=in_chans+1, embed_dim=embed_dims[0],
                    norm_layer=norm_layer, use_norm=ds_use_norm))
            else:
                self.patch_embeds.append(Downsample(embed_dims[i - 1], embed_dims[i],
                                                    patch_size=2, norm_layer=norm_layer,
                                                    use_norm=ds_use_norm))

            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], bias=bias,
                                 kernel_size=kernel_sizes[i], dilation=dilations[i], drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            self.layers.append(stage)

        self.head = SpadeDecoder(out_dim=in_chans, embed_dims=embed_dims,
                                 num_blks=(1, 3, 3, 4), downsample=downsample)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forword_backbone(self, x):
        feats = []
        for idx in range(self.num_stages):
            x = self.patch_embeds[idx](x)
            x = self.layers[idx](x)
            feats.append(x)
        return feats

    def forward(self, L, N):
        x = torch.cat((L, N), dim=1)
        feats = self.forword_backbone(x)
        out = self.head(feats, L)
        out = out + L
        return out



@register_model
def PhaseFormer_T(ds=4):
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    kernel_sizes = [11, 9, 7, 5]
    dilations = [2, 2, 2, 2]
    model = PhaseFormer(layers, embed_dims=embed_dims, bias=False,  in_chans=3,
                        kernel_sizes=kernel_sizes, dilations=dilations,
                        mlp_ratios=mlp_ratios, downsample=ds)
    model.default_cfg = default_cfgs['phaseformer_T']
    return model


@register_model
def PhaseFormer_S(ds=4):
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    kernel_sizes = [11, 9, 7, 5]
    dilations = [2, 2, 2, 2]
    model = PhaseFormer(layers, embed_dims=embed_dims, bias=False,  in_chans=3,
                        kernel_sizes=kernel_sizes, dilations=dilations,
                        mlp_ratios=mlp_ratios, downsample=ds)
    model.default_cfg = default_cfgs['phaseformer_S']
    return model


@register_model
def PhaseFormer_M(ds=4):
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    kernel_sizes = [11, 9, 7, 5]
    dilations = [2, 2, 2, 2]
    model = PhaseFormer(layers, embed_dims=embed_dims, bias=False,  in_chans=3,
                        kernel_sizes=kernel_sizes, dilations=dilations,
                        mlp_ratios=mlp_ratios, downsample=ds)
    model.default_cfg = default_cfgs['phaseformer_M']
    return model


@register_model
def PhaseFormer_B(ds=4):
    layers = [2, 2, 18, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    kernel_sizes = [11, 9, 7, 5]
    dilations = [2, 2, 2, 2]
    model = PhaseFormer(layers, embed_dims=embed_dims, bias=False,  in_chans=3,
                        kernel_sizes=kernel_sizes, dilations=dilations,
                        mlp_ratios=mlp_ratios, downsample=ds)
    model.default_cfg = default_cfgs['phaseformer_B']
    return model


def build_model(name, ds):
    if name == 'phaseformer_T':
        return PhaseFormer_T(ds)
    elif name == 'phaseformer_S':
        return PhaseFormer_S(ds)
    elif name == 'phaseformer_M':
        return PhaseFormer_M(ds)
    elif name == 'phaseformer_B':
        return PhaseFormer_B(ds)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    model = PhaseFormer_T(ds=3)
    L = torch.zeros(1, 3, 64, 64)
    N = torch.zeros(1, 1, 64, 64)
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Params: {:.2f}".format(params / 1e6))
    with torch.no_grad():
        y = model(L, N)
        print(y.shape)
