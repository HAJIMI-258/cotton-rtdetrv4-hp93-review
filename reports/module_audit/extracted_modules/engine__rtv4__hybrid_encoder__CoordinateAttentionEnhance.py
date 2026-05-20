# Exact source excerpt from engine/rtv4/hybrid_encoder.py:805-824
# Generated for module audit. Do not edit here; inspect original source for execution.

class CoordinateAttentionEnhance(nn.Module):
    """Coordinate attention keeps row/column position cues for leaf and boll structure."""

    def __init__(self, channels, reduction=32, alpha=0.12, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.reduce = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        context_h = x.mean(dim=3, keepdim=True)
        context_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.reduce(torch.cat([context_h, context_w], dim=2))
        context_h, context_w = torch.split(context, [h, w], dim=2)
        gate_h = torch.sigmoid(self.conv_h(context_h))
        gate_w = torch.sigmoid(self.conv_w(context_w.transpose(2, 3)))
        return x + self.alpha * x * gate_h * gate_w
