# Exact source excerpt from engine/rtv4/hybrid_encoder.py:461-476
# Generated for module audit. Do not edit here; inspect original source for execution.

class SmallLesionCrossScaleEnhance(nn.Module):
    """Lightweight cross-scale enhancement for high-resolution disease spots."""

    def __init__(self, channels, alpha=0.15, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.dw3 = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.dw5 = ConvNormLayer_fuse(channels, channels, 5, 1, g=channels, act=act)
        self.dw7 = ConvNormLayer_fuse(channels, channels, 7, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 3, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = self.mix(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        gate = torch.sigmoid(self.gate(context))
        return x + self.alpha * context * gate
