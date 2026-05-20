# Exact source excerpt from engine/rtv4/hybrid_encoder.py:760-773
# Generated for module audit. Do not edit here; inspect original source for execution.

class RepVGGEnhanceBlock(nn.Module):
    """RepVGG training-time 3x3 + 1x1 + identity block."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.branch_3x3 = ConvNormLayer(channels, channels, 3, 1, padding=1, act=None)
        self.branch_1x1 = ConvNormLayer(channels, channels, 1, 1, padding=0, act=None)
        self.branch_identity = nn.BatchNorm2d(channels)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        y = self.branch_3x3(x) + self.branch_1x1(x) + self.branch_identity(x)
        return x + self.alpha * self.act(y)
