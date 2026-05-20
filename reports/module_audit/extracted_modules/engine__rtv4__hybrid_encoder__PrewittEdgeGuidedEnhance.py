# Exact source excerpt from engine/rtv4/hybrid_encoder.py:220-259
# Generated for module audit. Do not edit here; inspect original source for execution.

class PrewittEdgeGuidedEnhance(nn.Module):
    """Prewitt edge prior for shallow feature enhancement.

    The fixed kernels follow the Prewitt operator templates:
        Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Gy = [[ 1, 1, 1], [ 0, 0, 0], [-1,-1,-1]]
    Gradient magnitude follows the paper description with infinity norm,
    and the adaptive threshold is T = k * Gmax.
    """

    def __init__(self, channels, alpha=0.15, threshold_ratio=0.2, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.eps = eps
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-1., 0., 1.],
             [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1., 1., 1.],
             [0., 0., 0.],
             [-1., -1., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('prewitt_kernel_x', kernel_x, persistent=False)
        self.register_buffer('prewitt_kernel_y', kernel_y, persistent=False)
        self.edge_proj = nn.Conv2d(1, channels, kernel_size=1, bias=True)

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)
        kernel_x = self.prewitt_kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.prewitt_kernel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(intensity, kernel_x, padding=1)
        gy = F.conv2d(intensity, kernel_y, padding=1)
        grad = torch.maximum(gx.abs(), gy.abs())
        gmax = grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        edge = grad * (grad >= self.threshold_ratio * gmax).to(grad.dtype)
        edge = edge / (gmax + self.eps)
        edge = torch.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0).detach()
        gate = torch.sigmoid(self.edge_proj(edge))
        return x * (1.0 + self.alpha * gate)
