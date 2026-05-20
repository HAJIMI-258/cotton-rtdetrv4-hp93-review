# Exact source excerpt from engine/rtv4/hybrid_encoder.py:726-739
# Generated for module audit. Do not edit here; inspect original source for execution.

class PartialConv2d(nn.Module):
    """FasterNet partial convolution on a channel subset."""

    def __init__(self, channels, n_div=4):
        super().__init__()
        self.dim_conv = channels // n_div
        self.dim_untouched = channels - self.dim_conv
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv, self.dim_conv, 3, 1, 1, bias=False
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat([self.partial_conv3(x1), x2], dim=1)
