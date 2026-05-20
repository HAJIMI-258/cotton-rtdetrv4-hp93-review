# Exact source excerpt from engine/rtv4/hybrid_encoder.py:697-723
# Generated for module audit. Do not edit here; inspect original source for execution.

class LSKBlock(nn.Module):
    """Large Selective Kernel block from LSKNet."""

    def __init__(self, channels, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, stride=1,
                                      padding=9, groups=channels, dilation=3)
        hidden = max(1, channels // 2)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn = torch.amax(attn, dim=1, keepdim=True)
        sig = torch.sigmoid(self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)))
        attn = attn1 * sig[:, 0:1] + attn2 * sig[:, 1:2]
        attn = self.conv(attn)
        return x + self.alpha * x * attn
