# Exact source excerpt from engine/rtv4/hybrid_encoder.py:500-514
# Generated for module audit. Do not edit here; inspect original source for execution.

class BackgroundSuppressionGate(nn.Module):
    """Foreground gate that weakens clutter influence through residual reweighting."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels + 2, 1, kernel_size=1, bias=True)

    def forward(self, x):
        local = self.local(x)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        fg_gate = torch.sigmoid(self.gate(torch.cat([local, avg_map, max_map], dim=1)))
        return x * (1.0 + self.alpha * fg_gate)
