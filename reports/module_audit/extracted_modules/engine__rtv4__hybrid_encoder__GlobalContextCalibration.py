# Exact source excerpt from engine/rtv4/hybrid_encoder.py:844-857
# Generated for module audit. Do not edit here; inspect original source for execution.

class GlobalContextCalibration(nn.Module):
    """SE-style global context calibration for field-background robustness."""

    def __init__(self, channels, reduction=16, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.fc1 = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(self.fc1(context)))
        return x + self.alpha * x * gate
