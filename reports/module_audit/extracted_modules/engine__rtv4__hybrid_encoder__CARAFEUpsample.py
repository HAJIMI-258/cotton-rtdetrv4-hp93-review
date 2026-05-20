# Exact source excerpt from engine/rtv4/hybrid_encoder.py:548-595
# Generated for module audit. Do not edit here; inspect original source for execution.

class CARAFEUpsample(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE).

    This follows the CARAFE operator: channel compression, content encoder,
    kernel normalization with pixel shuffle, and content-aware reassembly.
    The implementation is pure PyTorch so it does not depend on MMCV CUDA ops.
    """

    def __init__(self, channels, scale_factor=2, compressed_channels=64,
                 encoder_kernel=3, up_kernel=5, up_group=1, act='silu'):
        super().__init__()
        if up_group != 1:
            raise ValueError("Pure PyTorch CARAFE currently uses up_group=1.")
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.channel_compressor = ConvNormLayer(
            channels, compressed_channels, 1, 1, padding=0, act=act
        )
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            (up_kernel * up_kernel) * up_group * scale_factor * scale_factor,
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2,
            bias=True,
        )

    def forward(self, x, output_size=None):
        b, c, h, w = x.shape
        sf = self.scale_factor
        masks = self.content_encoder(self.channel_compressor(x))
        masks = F.pixel_shuffle(masks, sf)
        h_up, w_up = h * sf, w * sf

        if output_size is not None and (h_up, w_up) != tuple(output_size):
            masks = F.interpolate(masks, size=output_size, mode='bilinear', align_corners=False)
            h_up, w_up = output_size

        masks = masks.view(b, self.up_group, self.up_kernel * self.up_kernel, h_up, w_up)
        masks = F.softmax(masks, dim=2)

        x_up = F.interpolate(x, size=(h_up, w_up), mode='nearest')
        patches = F.unfold(
            x_up,
            kernel_size=self.up_kernel,
            padding=self.up_kernel // 2,
        ).view(b, c, self.up_kernel * self.up_kernel, h_up, w_up)
        return (patches * masks[:, 0:1]).sum(dim=2)
