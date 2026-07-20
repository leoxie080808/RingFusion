"""Network A -- the distilled monocular depth backbone (technical reference §3.3).

  Input   3 x 288 x 384   rectified RGB, ImageNet-normalized
  Encoder MobileNetV3-Large, ImageNet-pretrained, taps at strides 4/8/16/32
  Decoder DPT-lite: 4 fusion blocks, 64 channels, progressive x2 upsample + skips
  Head    2 conv layers -> 1 x 288 x 384, ReLU
  Output  relative disparity (non-negative), up to a global scale/shift

Only plain convolutions -- no attention, no deformable/dynamic ops -- because
those are the memory-bound operations that resist INT8 quantization on the Orin.
That constraint is *why* this runs where DELTAR does not, so keep it when editing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3Encoder(nn.Module):
    """Runs mobilenet_v3_large.features and returns the deepest feature map at
    each of strides (4, 8, 16, 32). Channel counts are probed at build time so
    the decoder wires up correctly without hard-coding block indices."""

    out_strides = (4, 8, 16, 32)

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.blocks = mobilenet_v3_large(weights=weights).features
        self.out_channels = self._probe()

    @torch.no_grad()
    def _probe(self):
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, 256, 256)
        chans = self._collect(x)
        self.train(was_training)
        return [chans[s][1] for s in self.out_strides]

    def _collect(self, x):
        """Map stride -> (feature tensor, channels), keeping the last block at
        each stride (largest receptive field before the next downsample)."""
        h0 = x.shape[-2]
        out = x
        collected = {}
        for blk in self.blocks:
            out = blk(out)
            s = h0 // out.shape[-2]
            collected[s] = (out, out.shape[1])
        return collected

    def forward(self, x):
        collected = self._collect(x)
        return [collected[s][0] for s in self.out_strides]


class ResidualConvUnit(nn.Module):
    """Pre-activation residual conv unit (the DPT building block)."""

    def __init__(self, ch):
        super().__init__()
        self.act = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.bn1(self.conv1(self.act(x)))
        y = self.bn2(self.conv2(self.act(y)))
        return x + y


class FusionBlock(nn.Module):
    """Fuse a skip feature with the (coarser) top-down path, then upsample x2."""

    def __init__(self, ch):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(ch)
        self.rcu_out = ResidualConvUnit(ch)

    def forward(self, skip, top_down=None):
        x = self.rcu_skip(skip)
        if top_down is not None:
            if top_down.shape[-2:] != x.shape[-2:]:
                top_down = F.interpolate(top_down, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)
            x = x + top_down
        x = self.rcu_out(x)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class DPTLiteDecoder(nn.Module):
    def __init__(self, in_channels, dim=64):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_channels])
        self.fusions = nn.ModuleList([FusionBlock(dim) for _ in in_channels])
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.ReLU(inplace=True),      # disparity is non-negative
        )

    def forward(self, feats, out_hw):
        p = None
        for i in reversed(range(len(feats))):          # stride 32 -> 4
            p = self.fusions[i](self.projs[i](feats[i]), p)
        disp = self.head(p)
        return F.interpolate(disp, size=out_hw, mode='bilinear', align_corners=False)


class DepthStudent(nn.Module):
    """Full student: encoder + DPT-lite decoder. forward(rgb) -> disparity."""

    def __init__(self, pretrained=True, decoder_dim=64):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained)
        self.decoder = DPTLiteDecoder(self.encoder.out_channels, decoder_dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.decoder(self.encoder(x), (h, w))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Shape + parameter-count check (no pretrained download).
    net = DepthStudent(pretrained=False)
    x = torch.zeros(2, 3, 288, 384)
    y = net(x)
    print(f"encoder tap channels: {net.encoder.out_channels}")
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"disparity non-negative: {bool((y >= 0).all())}")
    print(f"trainable parameters: {count_parameters(net):,}")
