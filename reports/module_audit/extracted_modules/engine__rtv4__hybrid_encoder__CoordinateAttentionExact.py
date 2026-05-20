# Exact source excerpt from engine/rtv4/hybrid_encoder.py:664-694
# Generated for module audit. Do not edit here; inspect original source for execution.

class CoordinateAttentionExact(nn.Module):
    """Coordinate Attention with separate height and width encodings."""

    def __init__(self, channels, reduction=32, alpha=0.12):
        super().__init__()
        self.alpha = alpha
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _h_sigmoid(x):
        return F.relu6(x + 3.0, inplace=True) / 6.0

    def _h_swish(self, x):
        return x * self._h_sigmoid(x)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self._h_swish(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity + self.alpha * identity * a_h * a_w
