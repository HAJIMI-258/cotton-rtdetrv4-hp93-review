# Exact source excerpt from engine/rtv4/hybrid_encoder.py:517-531
# Generated for module audit. Do not edit here; inspect original source for execution.

class ResidualContextRefine(nn.Module):
    """Residual multi-receptive-field refinement for each output scale."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.context = ConvNormLayer_fuse(channels, channels, 9, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        refined = self.mix(torch.cat([self.local(x), self.context(x)], dim=1))
        gate = torch.sigmoid(self.gate(refined))
        return x + self.alpha * refined * gate
