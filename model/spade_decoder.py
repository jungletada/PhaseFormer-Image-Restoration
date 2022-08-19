import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, upscale=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * (upscale ** 2), kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=upscale)
        )

    def forward(self, x):
        return self.upsample(x)


class SPADE(nn.Module):
    def __init__(self, dim, cond_dim=3, hidden_dim=128):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(dim, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
        )
        self.mlp_gamma = nn.Conv2d(hidden_dim, dim, kernel_size=3, stride=1, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, L):
        normalized = self.param_free_norm(x)
        L = F.interpolate(L, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(L)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class ResidualBlock(nn.Module):
    """
    return x + conv(GELU(conv(x)))
    """
    def __init__(self, dim=64, bias=False):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, x):
        return self.layers(x) + x


class SpadeResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim=3, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm = SPADE(dim=dim, cond_dim=cond_dim, hidden_dim=dim)
        self.actvn = nn.GELU()

    def forward(self, x, L):
        out = self.conv(x)
        out = self.norm(out, L)
        out = self.actvn(out)
        return out + x


class ConsecutiveResBlock(nn.Module):
    def __init__(self, dim, num_blks):
        super(ConsecutiveResBlock, self).__init__()
        self.layer_0 = SpadeResidualBlock(dim, cond_dim=3)
        self.layer_1 = nn.Sequential(*[ResidualBlock(dim) for _ in range(num_blks - 1)])

    def forward(self, x, L):
        out = self.layer_0(x, L)
        out = self.layer_1(out)
        return out + x


class SpadeDecoder(nn.Module):
    def __init__(self, out_dim=3, embed_dims=(64, 128, 320, 512),
                 num_blks=(1, 3, 3, 4), downsample=4):
        super(SpadeDecoder, self).__init__()
        self.up4 = Upsample(embed_dims[3], embed_dims[2], upscale=2)
        self.up3 = Upsample(embed_dims[2], embed_dims[1], upscale=2)
        self.up2 = Upsample(embed_dims[1], embed_dims[0], upscale=2)
        if downsample == 4:
            self.up1 = Upsample(embed_dims[0], embed_dims[0] // 2, upscale=2)
        elif downsample == 3:
            self.up1 = nn.Sequential(
                nn.Conv2d(embed_dims[0], embed_dims[0] // 2, kernel_size=3, stride=1, padding=1,bias=False),
                nn.GELU(),
            )
        self.resblk4 = ConsecutiveResBlock(embed_dims[3], num_blks=num_blks[0])
        self.resblk3 = ConsecutiveResBlock(embed_dims[2], num_blks=num_blks[1])
        self.resblk2 = ConsecutiveResBlock(embed_dims[1], num_blks=num_blks[2])
        self.resblk1 = ConsecutiveResBlock(embed_dims[0], num_blks=num_blks[3])

        self.tail = nn.Conv2d(embed_dims[0] // 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs, L):
        x1, x2, x3, x4 = inputs
        x = self.resblk4(x4, L) + x4
        x = self.up4(x) + x3
        x = self.resblk3(x, L)
        x = self.up3(x) + x2
        x = self.resblk2(x, L)
        x = self.up2(x) + x1
        x = self.resblk1(x, L)
        x = self.up1(x)
        x = self.tail(x)
        return x


if __name__ == '__main__':
    dim = 64
    L = torch.zeros(1, 3, 128, 128)
    x1 = torch.zeros(1, dim, 64, 64)
    x2 = torch.zeros(1, dim*2, 32, 32)
    x3 = torch.zeros(1, dim*4, 16, 16)
    x4 = torch.zeros(1, dim*8, 8, 8)
    inputs = (x1, x2, x3, x4)
    model = SpadeDecoder(embed_dims=(64, 128, 256, 512),)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    print("Params: {:.2f}".format(params / 1e6))
    with torch.no_grad():
        y = model(inputs, L)
        print(y.shape)