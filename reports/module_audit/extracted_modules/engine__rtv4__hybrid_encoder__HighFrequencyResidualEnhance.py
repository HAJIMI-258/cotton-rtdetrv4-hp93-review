# Exact source excerpt from engine/rtv4/hybrid_encoder.py:827-841
# Generated for module audit. Do not edit here; inspect original source for execution.

class HighFrequencyResidualEnhance(nn.Module):
    """High-frequency residual enhancement for tiny leaf spots and pest edges."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.proj = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        enhanced = self.proj(high)
        gate = torch.sigmoid(self.gate(high))
        return x + self.alpha * enhanced * gate
