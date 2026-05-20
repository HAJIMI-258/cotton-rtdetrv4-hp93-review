# Exact source excerpt from engine/rtv4/hybrid_encoder.py:479-497
# Generated for module audit. Do not edit here; inspect original source for execution.

class EfficientChannelSpatialAttention(nn.Module):
    """Lightweight channel-spatial attention for weak texture and background suppression."""

    def __init__(self, channels, alpha=0.15, spatial_kernel_size=7):
        super().__init__()
        self.alpha = alpha
        padding = (spatial_kernel_size - 1) // 2
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=padding, bias=False)

    def forward(self, x):
        channel_context = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        channel_gate = torch.sigmoid(self.channel_conv(channel_context).transpose(1, 2).unsqueeze(-1))

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1)))

        return x + self.alpha * x * channel_gate * spatial_gate
