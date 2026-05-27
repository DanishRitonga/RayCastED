import torch.nn as nn
from torchvision.ops import DeformConv2d


class DCNConv(nn.Module):
    """Modulated Deformable Conv (DCNv2). Drop-in replacement for Conv(c1, c2, 3).

    Zero-init offsets+mask -> starts as standard conv at init.
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_conv = nn.Conv2d(
            c1,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.mask_conv = nn.Conv2d(
            c1,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        self.dcn = DeformConv2d(
            c1,
            c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = self.mask_conv(x).sigmoid()
        x = self.dcn(x, offset, mask)
        return self.act(self.bn(x))
