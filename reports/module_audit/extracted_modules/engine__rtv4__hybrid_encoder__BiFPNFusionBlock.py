# Exact source excerpt from engine/rtv4/hybrid_encoder.py:598-614
# Generated for module audit. Do not edit here; inspect original source for execution.

class BiFPNFusionBlock(nn.Module):
    """EfficientDet BiFPN normalized weighted fusion plus separable conv."""

    def __init__(self, channels, num_inputs=2, eps=1e-4, act='silu'):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.dwconv = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=None)
        self.pwconv = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)

    def forward(self, *inputs):
        if len(inputs) != self.weights.numel():
            raise ValueError(f"BiFPNFusionBlock expects {self.weights.numel()} inputs, got {len(inputs)}.")
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = sum(x * weights[i] for i, x in enumerate(inputs))
        return self.pwconv(self.dwconv(fused))
