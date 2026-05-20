# Exact source excerpt from engine/rtv4/hybrid_encoder.py:636-661
# Generated for module audit. Do not edit here; inspect original source for execution.

class CBAMLayer(nn.Module):
    """CBAM channel and spatial attention."""

    def __init__(self, channels, reduction=16, spatial_kernel=7, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False
        )

    def forward(self, x):
        avg_attn = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.mlp(F.adaptive_max_pool2d(x, 1))
        channel_gate = torch.sigmoid(avg_attn + max_attn)
        y = x * channel_gate
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out = torch.amax(y, dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        y = y * spatial_gate
        return x + self.alpha * y
