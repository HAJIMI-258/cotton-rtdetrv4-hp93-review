# Exact source excerpt from engine/rtv4/hybrid_encoder.py:617-633
# Generated for module audit. Do not edit here; inspect original source for execution.

class ECALayer(nn.Module):
    """ECA-Net channel attention with adaptive 1D kernel size."""

    def __init__(self, channels, gamma=2, b=1, alpha=0.10):
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(3, kernel_size)
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y).transpose(1, 2).unsqueeze(-1))
        return x + self.alpha * x * y
