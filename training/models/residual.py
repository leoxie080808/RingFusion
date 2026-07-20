"""Network B -- the residual correction net g_phi (technical reference §4.3).

  Input (6 channels, full resolution):
    [0:3]  rectified RGB, normalized
    [3]    D0 anchored depth, log-normalized: log(D0 / median)
    [4]    sparse anchor depth (ToF zones splatted to projected pixels, 0 else)
    [5]    anchor validity mask (1 where a zone landed)
  Encoder 32 -> 64 -> 96, stride-2 convs, 2 blocks each
  Decoder skip connections, bilinear upsample + conv
  Head    1x1 conv -> 3 channels: da, db, log(tau^2)
  Init    final conv weight = 0 AND bias = 0

Channels 4 and 5 tell the net *where* the anchors were -- error grows with
distance from an anchor, and the mask makes that relationship learnable.

The zero-initialized head is load-bearing: at step 0, da = db = 0, so
D = 1 / ((a)*disp + (b)) is exactly the closed-form fit -- g_phi is the identity
and training can only improve on the analytic solution. Keep it zero-init.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    """Two 3x3 conv-BN-ReLU layers."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ResidualRefinerNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, widths=(32, 64, 96)):
        super().__init__()
        c1, c2, c3 = widths
        self.enc1 = conv_block(in_ch, c1)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.enc2 = conv_block(c1, c2)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.bottleneck = conv_block(c2, c3)

        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec2 = conv_block(c2 + c2, c2)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec1 = conv_block(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.enc1(x)                              # full res
        e2 = self.enc2(self.down1(e1))                 # 1/2
        b = self.bottleneck(self.down2(e2))            # 1/4

        d2 = F.interpolate(b, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))
        return self.head(d1)                           # (B,3,H,W)


def split_output(out):
    """(B,3,H,W) -> da, db, log_tau2 each (B,1,H,W)."""
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def apply_residual(disp, a, b, out):
    """Compose the residual with the closed-form fit.
    disp: (B,1,H,W); a,b: (B,1,1,1) or scalars; out: head output (B,3,H,W).
    Returns metric depth D (B,1,H,W) and extra variance tau2 (B,1,H,W)."""
    da, db, log_tau2 = split_output(out)
    inv = (a + da) * disp + (b + db)
    D = 1.0 / inv.clamp(min=1e-4)
    return D, log_tau2.exp()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = ResidualRefinerNet()
    x = torch.zeros(2, 6, 288, 384)
    out = net(x)
    da, db, log_tau2 = split_output(out)
    print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}")
    print(f"zero-init identity: da==0 {bool((da == 0).all())}, "
          f"db==0 {bool((db == 0).all())}, log_tau2==0 {bool((log_tau2 == 0).all())}")
    print(f"trainable parameters: {count_parameters(net):,}")
