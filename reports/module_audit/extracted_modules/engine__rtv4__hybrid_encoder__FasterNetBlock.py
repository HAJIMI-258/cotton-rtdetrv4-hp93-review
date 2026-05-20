# Exact source excerpt from engine/rtv4/hybrid_encoder.py:742-757
# Generated for module audit. Do not edit here; inspect original source for execution.

class FasterNetBlock(nn.Module):
    """FasterNet block with partial convolution and pointwise MLP."""

    def __init__(self, channels, mlp_ratio=2.0, n_div=4, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        hidden = int(channels * mlp_ratio)
        self.spatial_mixing = PartialConv2d(channels, n_div=n_div)
        self.mlp = nn.Sequential(
            ConvNormLayer_fuse(channels, hidden, 1, 1, act=act),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.alpha * self.mlp(self.spatial_mixing(x))
