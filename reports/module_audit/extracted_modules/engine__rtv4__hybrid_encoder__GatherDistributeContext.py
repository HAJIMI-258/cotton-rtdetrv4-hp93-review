# Exact source excerpt from engine/rtv4/hybrid_encoder.py:776-802
# Generated for module audit. Do not edit here; inspect original source for execution.

class GatherDistributeContext(nn.Module):
    """Multi-scale gather-distribute context injection for neck outputs.

    The module gathers all neck outputs at each target scale, fuses them with a
    1x1 projection, and injects the distributed context residually.
    """

    def __init__(self, channels, num_levels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * num_levels, channels, 1, 1, act=act)
            for _ in range(num_levels)
        ])

    def forward(self, feats):
        outs = []
        for i, feat in enumerate(feats):
            size = feat.shape[-2:]
            aligned = [
                src if src.shape[-2:] == size
                else F.interpolate(src, size=size, mode='bilinear', align_corners=False)
                for src in feats
            ]
            context = self.fuse[i](torch.cat(aligned, dim=1))
            outs.append(feat + self.alpha * context)
        return outs
