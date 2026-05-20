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


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:262-458
# Generated for module audit. Do not edit here; inspect original source for execution.

class PrewittFranklinEdgeGuidedEnhance(nn.Module):
    """Prewitt-Franklin edge gate following the paper's full decision flow.

    Implemented steps:
        1. Prewitt pixel-level coarse edge extraction with infinity norm.
        2. T = k * Gmax with k = 0.2.
        3. 7x7 Franklin moment templates M00, M11, M20, M31 and M40.
        4. phi, l1, l2, l and k from Eq. (15), (17)-(20).
        5. Edge decision kt < k, lt < l and |l1-l2| < lt from Eq. (23).

    The Franklin templates are sampled from the Franklin radial basis functions
    phi_0 ... phi_4 given in Eq. (3)-(7), combined with the polar moment form
    in Eq. (9). The module uses the resulting decision map as a fixed
    image-processing prior and learns only the 1x1 projection into feature
    channels.
    """

    def __init__(
        self,
        channels,
        alpha=0.15,
        threshold_ratio=0.2,
        kernel_size=7,
        local_refine=True,
        local_refine_kernel_size=7,
        edge_consistency=False,
        eps=1e-6,
        act='silu',
    ):
        super().__init__()
        if kernel_size != 7:
            raise ValueError("The paper-selected Franklin template size is 7x7.")
        if local_refine_kernel_size % 2 != 1:
            raise ValueError("local_refine_kernel_size must be odd.")
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.edge_consistency = edge_consistency
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
        self.register_buffer('franklin_kernels', self._build_franklin_kernels(kernel_size), persistent=False)
        self.edge_proj = nn.Conv2d(4, channels, kernel_size=1, bias=True)
        self.feature_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.local_refine = nn.Sequential(
            ConvNormLayer_fuse(
                channels,
                channels,
                local_refine_kernel_size,
                1,
                g=channels,
                act=act,
            ),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        ) if local_refine else None
        self.edge_consistency_loss = None

    @staticmethod
    def _franklin_phi(order, r):
        if order == 0:
            return torch.ones_like(r)
        if order == 1:
            return (3.0 ** 0.5) * (2.0 * r - 1.0)
        if order == 2:
            return torch.where(
                r < 0.5,
                (3.0 ** 0.5) * (1.0 - 4.0 * r),
                (3.0 ** 0.5) * (4.0 * r - 3.0),
            )
        if order == 3:
            scale = (33.0 ** 0.5) / 11.0
            return torch.where(
                r < 0.25,
                scale * (5.0 - 38.0 * r),
                torch.where(
                    r < 0.5,
                    scale * (26.0 * r - 11.0),
                    scale * (5.0 - 6.0 * r),
                ),
            )
        if order == 4:
            scale = (231.0 ** 0.5) / 77.0
            return torch.where(
                r < 0.25,
                scale * (1.0 - 12.0 * r),
                torch.where(
                    r < 0.5,
                    scale * (36.0 * r - 11.0),
                    torch.where(
                        r < 0.75,
                        scale * (45.0 - 76.0 * r),
                        scale * (100.0 * r - 87.0),
                    ),
                ),
            )
        raise ValueError(f"Unsupported Franklin order: {order}")

    @classmethod
    def _build_franklin_kernels(cls, kernel_size):
        coords = torch.linspace(-1.0, 1.0, kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        rr = torch.sqrt(xx * xx + yy * yy).clamp(max=1.0)
        theta = torch.atan2(yy, xx)
        disk = (xx * xx + yy * yy <= 1.0).to(torch.float32)
        area = (2.0 / (kernel_size - 1)) ** 2

        def moment_kernel(order, angular_order, part='real'):
            radial = cls._franklin_phi(order, rr)
            scale = (order + 1.0) / torch.pi
            if angular_order == 0:
                angular = torch.ones_like(theta)
            elif part == 'real':
                angular = torch.cos(angular_order * theta)
            else:
                angular = -torch.sin(angular_order * theta)
            return scale * radial * angular * disk * area

        kernels = torch.stack([
            moment_kernel(0, 0),
            moment_kernel(1, 1, 'real'),
            moment_kernel(1, 1, 'imag'),
            moment_kernel(2, 0),
            moment_kernel(3, 1, 'real'),
            moment_kernel(3, 1, 'imag'),
            moment_kernel(4, 0),
        ], dim=0).unsqueeze(1)
        return kernels

    def _safe_div(self, num, den):
        den_safe = torch.where(den >= 0, torch.full_like(den, self.eps), torch.full_like(den, -self.eps))
        den = torch.where(den.abs() < self.eps, den_safe, den)
        return num / den

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)

        kernel_x = self.prewitt_kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.prewitt_kernel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(intensity, kernel_x, padding=1)
        gy = F.conv2d(intensity, kernel_y, padding=1)
        grad = torch.maximum(gx.abs(), gy.abs())
        gmax = grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        prewitt_edge = (grad >= self.threshold_ratio * gmax).to(grad.dtype)
        grad_norm = grad / (gmax + self.eps)

        kernels = self.franklin_kernels.to(device=x.device, dtype=x.dtype)
        moments = F.conv2d(intensity, kernels, padding=3)
        f00, f11_re, f11_im, f20, f31_re, f31_im, f40 = moments.chunk(7, dim=1)
        phi = torch.atan2(f11_im, f11_re + self.eps)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        f11_rot = f11_re * cos_phi + f11_im * sin_phi
        f31_rot = f31_re * cos_phi + f31_im * sin_phi
        f20_rot = f20
        f40_rot = f40

        l1_arg = self._safe_div(5.0 * f40_rot + 3.0 * f20_rot, 8.0 * f20_rot)
        l2_arg = self._safe_div(5.0 * f31_rot + 3.0 * f20_rot, 8.0 * f20_rot)
        l1 = l1_arg.clamp(min=self.eps, max=1.0).sqrt()
        l2 = l2_arg.clamp(min=self.eps, max=1.0).sqrt()
        l = ((l1 + l2) * 0.5).clamp(min=0.0, max=1.0 - self.eps)

        denom = 2.0 * (1.0 - l * l).clamp(min=self.eps).pow(1.5)
        k = (3.0 * f11_rot.abs()) / (denom + self.eps)
        lt = l.flatten(2).mean(dim=2).view(-1, 1, 1, 1)
        kt = k.flatten(2).mean(dim=2).view(-1, 1, 1, 1)
        franklin_edge = ((kt < k) & (lt < l) & ((l1 - l2).abs() < lt)).to(x.dtype)
        franklin_edge = franklin_edge * prewitt_edge
        k_norm = k / (k.flatten(1).amax(dim=1).view(-1, 1, 1, 1) + self.eps)

        edge_features = torch.cat([grad_norm, franklin_edge, l, k_norm], dim=1)
        edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=0.0)
        edge_prior = (0.65 * franklin_edge + 0.35 * grad_norm).clamp(0.0, 1.0).detach()
        edge_features = edge_features.detach()
        gate = torch.sigmoid(self.edge_proj(edge_features) + self.feature_gate(x))

        if self.local_refine is not None:
            refined = self.local_refine(x)
            delta = refined - x
        else:
            delta = x * gate

        if self.edge_consistency:
            gate_map = gate.mean(dim=1, keepdim=True)
            self.edge_consistency_loss = F.smooth_l1_loss(gate_map, edge_prior, reduction='mean')
        else:
            self.edge_consistency_loss = None

        return x + self.alpha * gate * delta


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:461-476
# Generated for module audit. Do not edit here; inspect original source for execution.

class SmallLesionCrossScaleEnhance(nn.Module):
    """Lightweight cross-scale enhancement for high-resolution disease spots."""

    def __init__(self, channels, alpha=0.15, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.dw3 = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.dw5 = ConvNormLayer_fuse(channels, channels, 5, 1, g=channels, act=act)
        self.dw7 = ConvNormLayer_fuse(channels, channels, 7, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 3, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = self.mix(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        gate = torch.sigmoid(self.gate(context))
        return x + self.alpha * context * gate


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


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:500-514
# Generated for module audit. Do not edit here; inspect original source for execution.

class BackgroundSuppressionGate(nn.Module):
    """Foreground gate that weakens clutter influence through residual reweighting."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels + 2, 1, kernel_size=1, bias=True)

    def forward(self, x):
        local = self.local(x)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        fg_gate = torch.sigmoid(self.gate(torch.cat([local, avg_map, max_map], dim=1)))
        return x * (1.0 + self.alpha * fg_gate)


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:517-531
# Generated for module audit. Do not edit here; inspect original source for execution.

class ResidualContextRefine(nn.Module):
    """Residual multi-receptive-field refinement for each output scale."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.context = ConvNormLayer_fuse(channels, channels, 9, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        refined = self.mix(torch.cat([self.local(x), self.context(x)], dim=1))
        gate = torch.sigmoid(self.gate(refined))
        return x + self.alpha * refined * gate


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


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:742-757
# Generated for module audit. Do not edit here; inspect original source for execution.

class FasterNetBlock(nn.Module):
    """FasterNet block with partial convolution and pointwise MLP."""

    def __init__(self, channels, mlp_ratio=2.0, n_div=4, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        hidden = int(channels * mlp_ratio)
        self.spatial_mixing = PartialConv2d(channels, n_div=n_div)
        self.mlp = nn.Sequential(
            ConvNormLayer_fuse(channels, hidden, 1, 1, act=act),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.alpha * self.mlp(self.spatial_mixing(x))


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:760-773
# Generated for module audit. Do not edit here; inspect original source for execution.

class RepVGGEnhanceBlock(nn.Module):
    """RepVGG training-time 3x3 + 1x1 + identity block."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.branch_3x3 = ConvNormLayer(channels, channels, 3, 1, padding=1, act=None)
        self.branch_1x1 = ConvNormLayer(channels, channels, 1, 1, padding=0, act=None)
        self.branch_identity = nn.BatchNorm2d(channels)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        y = self.branch_3x3(x) + self.branch_1x1(x) + self.branch_identity(x)
        return x + self.alpha * self.act(y)


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


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:805-824
# Generated for module audit. Do not edit here; inspect original source for execution.

class CoordinateAttentionEnhance(nn.Module):
    """Coordinate attention keeps row/column position cues for leaf and boll structure."""

    def __init__(self, channels, reduction=32, alpha=0.12, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.reduce = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        context_h = x.mean(dim=3, keepdim=True)
        context_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.reduce(torch.cat([context_h, context_w], dim=2))
        context_h, context_w = torch.split(context, [h, w], dim=2)
        gate_h = torch.sigmoid(self.conv_h(context_h))
        gate_w = torch.sigmoid(self.conv_w(context_w.transpose(2, 3)))
        return x + self.alpha * x * gate_h * gate_w


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:827-841
# Generated for module audit. Do not edit here; inspect original source for execution.

class HighFrequencyResidualEnhance(nn.Module):
    """High-frequency residual enhancement for tiny leaf spots and pest edges."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.proj = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        enhanced = self.proj(high)
        gate = torch.sigmoid(self.gate(high))
        return x + self.alpha * enhanced * gate


# Exact source excerpt from engine/rtv4/hybrid_encoder.py:844-857
# Generated for module audit. Do not edit here; inspect original source for execution.

class GlobalContextCalibration(nn.Module):
    """SE-style global context calibration for field-background robustness."""

    def __init__(self, channels, reduction=16, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.fc1 = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(self.fc1(context)))
        return x + self.alpha * x * gate


# Exact source excerpt from engine/data/transforms/copy_paste.py:21-220
# Generated for module audit. Do not edit here; inspect original source for execution.

class HardClassCopyPaste(T.Transform):
    def __init__(
        self,
        target_category_names: Optional[Iterable[str]] = None,
        target_category_ids: Optional[Iterable[int]] = None,
        probability: float = 0.35,
        max_paste: int = 2,
        max_trials: int = 25,
        context_ratio: float = 0.18,
        scale_range=(0.90, 1.10),
        max_patch_ratio: float = 0.48,
        max_overlap_iou: float = 0.35,
        min_box_size: int = 4,
        max_epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target_category_names = list(target_category_names or [])
        self.target_category_ids = set(int(x) for x in (target_category_ids or []))
        self.probability = float(probability)
        self.max_paste = int(max_paste)
        self.max_trials = int(max_trials)
        self.context_ratio = float(context_ratio)
        self.scale_range = tuple(float(x) for x in scale_range)
        self.max_patch_ratio = float(max_patch_ratio)
        self.max_overlap_iou = float(max_overlap_iou)
        self.min_box_size = int(min_box_size)
        self.max_epoch = max_epoch
        self._cached_dataset_id = None
        self._resolved_ids = set()
        self._warned_missing = False

    def _resolve_target_ids(self, dataset):
        dataset_id = id(dataset)
        if dataset_id == self._cached_dataset_id:
            return self._resolved_ids

        ids = set(self.target_category_ids)
        name_to_id = {}
        for cat in getattr(dataset, 'categories', []) or []:
            name_to_id[str(cat.get('name'))] = int(cat.get('id'))

        missing = []
        for name in self.target_category_names:
            if name in name_to_id:
                ids.add(name_to_id[name])
            else:
                missing.append(name)

        if missing and not self._warned_missing:
            print(f"     ### HardClassCopyPaste missing categories skipped: {missing} ###")
            self._warned_missing = True

        self._cached_dataset_id = dataset_id
        self._resolved_ids = ids
        return ids

    @staticmethod
    def _clone_target(target):
        cloned = {}
        for key, value in target.items():
            cloned[key] = value.clone() if torch.is_tensor(value) else value
        return cloned

    @staticmethod
    def _box_iou_one_to_many(box, boxes):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        lt = torch.maximum(box[:2], boxes[:, :2])
        rb = torch.minimum(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return inter / (area1 + area2 - inter).clamp(min=1e-6)

    def _sample_donor(self, dataset, target_ids):
        for _ in range(self.max_trials):
            idx = random.randrange(len(dataset))
            donor_img, donor_target = dataset.load_item(idx)
            labels = donor_target.get('labels')
            boxes = donor_target.get('boxes')
            if labels is None or boxes is None or len(labels) == 0:
                continue
            mask = torch.tensor([int(label) in target_ids for label in labels], dtype=torch.bool)
            if not mask.any():
                continue
            candidates = mask.nonzero(as_tuple=False).flatten().tolist()
            obj_idx = random.choice(candidates)
            return donor_img, donor_target, obj_idx
        return None

    def _build_patch(self, donor_img, donor_target, obj_idx, base_size):
        if not isinstance(donor_img, Image.Image):
            return None

        donor_w, donor_h = donor_img.size
        boxes = torch.as_tensor(donor_target['boxes'], dtype=torch.float32)
        box = boxes[obj_idx].clone()
        x1, y1, x2, y2 = box.tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw < self.min_box_size or bh < self.min_box_size:
            return None

        context = self.context_ratio * max(bw, bh)
        cx1 = max(0, int(round(x1 - context)))
        cy1 = max(0, int(round(y1 - context)))
        cx2 = min(donor_w, int(round(x2 + context)))
        cy2 = min(donor_h, int(round(y2 + context)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = donor_img.crop((cx1, cy1, cx2, cy2)).convert(donor_img.mode)
        rel_box = torch.tensor([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=torch.float32)
        scale = random.uniform(*self.scale_range)

        base_w, base_h = base_size
        max_pw = max(1, int(base_w * self.max_patch_ratio))
        max_ph = max(1, int(base_h * self.max_patch_ratio))
        new_w = max(1, int(round(crop.size[0] * scale)))
        new_h = max(1, int(round(crop.size[1] * scale)))
        fit_scale = min(1.0, max_pw / max(new_w, 1), max_ph / max(new_h, 1))
        new_w = max(1, int(round(new_w * fit_scale)))
        new_h = max(1, int(round(new_h * fit_scale)))
        if new_w >= base_w or new_h >= base_h:
            return None

        sx = new_w / max(crop.size[0], 1)
        sy = new_h / max(crop.size[1], 1)
        crop = crop.resize((new_w, new_h), Image.BILINEAR)
        rel_box = rel_box * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        label = donor_target['labels'][obj_idx].view(1).clone()
        return crop, rel_box, label

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        if self.max_epoch is not None and getattr(dataset, 'epoch', 0) >= self.max_epoch:
            return image, target, dataset
        if self.probability <= 0 or random.random() > self.probability:
            return image, target, dataset
        if not isinstance(image, Image.Image):
            return image, target, dataset

        target_ids = self._resolve_target_ids(dataset)
        if not target_ids:
            return image, target, dataset

        out_img = image.copy()
        out_target = self._clone_target(target)
        base_w, base_h = out_img.size
        existing_boxes = torch.as_tensor(out_target.get('boxes', torch.empty((0, 4))), dtype=torch.float32)
        paste_count = random.randint(1, max(1, self.max_paste))

        new_boxes, new_labels, new_areas, new_iscrowd = [], [], [], []
        for _ in range(paste_count):
            donor = self._sample_donor(dataset, target_ids)
            if donor is None:
                break
            patch = self._build_patch(*donor, base_size=(base_w, base_h))
            if patch is None:
                continue
            crop, rel_box, label = patch
            crop_w, crop_h = crop.size

            placed = False
            for _trial in range(20):
                px = random.randint(0, max(0, base_w - crop_w))
                py = random.randint(0, max(0, base_h - crop_h))
                new_box = rel_box + torch.tensor([px, py, px, py], dtype=torch.float32)
                if (new_box[2] - new_box[0]) < self.min_box_size or (new_box[3] - new_box[1]) < self.min_box_size:
                    continue
                if existing_boxes.numel() and self._box_iou_one_to_many(new_box, existing_boxes).max().item() > self.max_overlap_iou:
                    continue
                out_img.paste(crop, (px, py))
                existing_boxes = torch.cat([existing_boxes, new_box.view(1, 4)], dim=0)
                new_boxes.append(new_box.view(1, 4))
                new_labels.append(label)
                new_areas.append(((new_box[2] - new_box[0]) * (new_box[3] - new_box[1])).view(1))
                new_iscrowd.append(torch.zeros(1, dtype=torch.int64))
                placed = True
                break
            if not placed:
                continue

        if new_boxes:
            out_target['boxes'] = convert_to_tv_tensor(
                torch.cat([torch.as_tensor(out_target['boxes'], dtype=torch.float32)] + new_boxes, dim=0),
                key='boxes',
                box_format='xyxy',
                spatial_size=out_img.size[::-1],
            )
            out_target['labels'] = torch.cat([out_target['labels']] + new_labels, dim=0)
            if 'area' in out_target:
                out_target['area'] = torch.cat([out_target['area'].to(torch.float32)] + new_areas, dim=0)
            if 'iscrowd' in out_target:
                out_target['iscrowd'] = torch.cat([out_target['iscrowd'].to(torch.int64)] + new_iscrowd, dim=0)

        return out_img, out_target, dataset


# Exact source excerpt from engine/rtv4/rtv4_criterion.py:28-696
# Generated for module audit. Do not edit here; inspect original source for execution.

class RTv4Criterion(nn.Module):
    """ This class computes the loss for RT-DETRv4.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True,
                 nwd_normalizer=1.0,
                 nwd_eps=1e-7,
                 hdps_class_union=False,
                 distill_adaptive_params=None,
                 class_loss_weights=None,
                 ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.nwd_normalizer = nwd_normalizer
        self.nwd_eps = nwd_eps
        self.hdps_class_union = hdps_class_union

        self.distill_adaptive_params = distill_adaptive_params
        class_weight_tensor = torch.ones(num_classes, dtype=torch.float32)
        for class_id, weight in (class_loss_weights or {}).items():
            try:
                class_id = int(class_id)
                weight = float(weight)
            except (TypeError, ValueError):
                _logger.warning("Ignore non-numeric class loss weight key: %s", class_id)
                continue
            if 0 <= class_id < num_classes:
                class_weight_tensor[class_id] = weight
            else:
                _logger.warning(
                    "Ignore class loss weight for id %s outside [0, %s).",
                    class_id,
                    num_classes,
                )
        self.register_buffer('class_loss_weight_tensor', class_weight_tensor, persistent=False)

    def normalized_gaussian_wasserstein_similarity(self, src_boxes, target_boxes):
        """NWD similarity for boxes represented as normalized cxcywh.

        It models each box as a 2D Gaussian and uses the closed-form Wasserstein
        distance: ||center1-center2||^2 + ((w1-w2)^2 + (h1-h2)^2) / 4.
        """
        center_distance = (src_boxes[:, :2] - target_boxes[:, :2]).pow(2).sum(dim=-1)
        wh_distance = (src_boxes[:, 2:] - target_boxes[:, 2:]).pow(2).sum(dim=-1) / 4.0
        wasserstein = torch.sqrt(center_distance + wh_distance + self.nwd_eps)
        return torch.exp(-wasserstein / self.nwd_normalizer)


    def loss_distillation(self, outputs, targets, indices, num_boxes, **kwargs):
        """Foreground-aware feature distillation.

        The original implementation averaged cosine distance over every spatial
        token. For cotton disease/pest images, the target occupies a small part
        of the image, so a full-map average can over-regularize the student to
        learn leaf/background texture rather than lesion/pest regions.

        This version keeps the same DINOv3 teacher/student feature interface but
        weights feature tokens inside enlarged GT boxes much higher than the
        background. It is controlled through distill_adaptive_params:
            foreground_focus: bool, default True
            fg_weight: float, default 1.0
            bg_weight: float, default 0.03
            box_scale: float, default 1.8
            min_fg_pixels: int, default 4
        """
        student_feature_map = outputs.get('student_distill_output')
        teacher_feature_map = outputs.get('teacher_encoder_output')

        if student_feature_map is None or teacher_feature_map is None:
            ref = outputs.get('pred_logits')
            device = ref.device if torch.is_tensor(ref) else torch.device('cpu')
            return {'loss_distill': torch.zeros((), device=device)}

        student_feature_map = student_feature_map.float()
        teacher_feature_map = teacher_feature_map.detach().float()

        if student_feature_map.shape[1] != teacher_feature_map.shape[1]:
            _logger.error(
                f"[RTv4Criterion] Feature dimension mismatch! Student: {student_feature_map.shape[1]}, Teacher: {teacher_feature_map.shape[1]}")
            raise ValueError("Feature dimension mismatch between student and teacher for distillation loss.")

        H_s, W_s = student_feature_map.shape[2:]
        H_t, W_t = teacher_feature_map.shape[2:]

        if (H_s, W_s) != (H_t, W_t):
            _logger.warning(
                f"[RTv4Criterion] Resizing teacher feature map from {H_t}x{W_t} to student's {H_s}x{W_s} for distillation.")
            teacher_feature_map = F.interpolate(
                teacher_feature_map, size=(H_s, W_s), mode='bilinear', align_corners=False)

        student_output_flat = student_feature_map.flatten(2).permute(0, 2, 1)
        teacher_output_flat = teacher_feature_map.flatten(2).permute(0, 2, 1)

        student_output_norm = F.normalize(student_output_flat, p=2, dim=-1, eps=1e-6)
        teacher_output_norm = F.normalize(teacher_output_flat, p=2, dim=-1, eps=1e-6)

        cos_sim = (student_output_norm * teacher_output_norm).sum(dim=-1).clamp(min=-1.0, max=1.0)
        loss_map = 1.0 - cos_sim

        params = self.distill_adaptive_params or {}
        foreground_focus = params.get('foreground_focus', True)

        if foreground_focus and targets is not None and len(targets) > 0:
            bg_weight = float(params.get('bg_weight', 0.03))
            fg_weight = float(params.get('fg_weight', 1.0))
            box_scale = float(params.get('box_scale', 1.8))
            min_fg_pixels = int(params.get('min_fg_pixels', 4))

            B = loss_map.shape[0]
            weights_map = torch.full(
                (B, H_s, W_s), bg_weight, dtype=loss_map.dtype, device=loss_map.device)

            for b, target in enumerate(targets[:B]):
                boxes = target.get('boxes', None)
                if boxes is None or boxes.numel() == 0:
                    continue
                boxes = boxes.to(device=loss_map.device, dtype=loss_map.dtype)
                cx, cy, bw, bh = boxes.unbind(dim=1)
                bw = (bw * box_scale).clamp(min=1.0 / max(W_s, 1), max=1.0)
                bh = (bh * box_scale).clamp(min=1.0 / max(H_s, 1), max=1.0)

                x1 = ((cx - bw * 0.5) * W_s).floor().clamp(0, W_s - 1).long()
                y1 = ((cy - bh * 0.5) * H_s).floor().clamp(0, H_s - 1).long()
                x2 = ((cx + bw * 0.5) * W_s).ceil().clamp(1, W_s).long()
                y2 = ((cy + bh * 0.5) * H_s).ceil().clamp(1, H_s).long()

                for _x1, _y1, _x2, _y2 in zip(x1, y1, x2, y2):
                    if (_x2 - _x1) * (_y2 - _y1) < min_fg_pixels:
                        cx_i = ((_x1 + _x2) // 2).clamp(0, W_s - 1)
                        cy_i = ((_y1 + _y2) // 2).clamp(0, H_s - 1)
                        half = max(1, int(min_fg_pixels ** 0.5))
                        _x1 = torch.clamp(cx_i - half, 0, W_s - 1)
                        _x2 = torch.clamp(cx_i + half + 1, 1, W_s)
                        _y1 = torch.clamp(cy_i - half, 0, H_s - 1)
                        _y2 = torch.clamp(cy_i + half + 1, 1, H_s)
                    weights_map[b, _y1:_y2, _x1:_x2] = fg_weight

            weights = weights_map.flatten(1)
            loss_distill = (loss_map * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            loss_distill = loss_map.mean()

        loss_distill = torch.nan_to_num(loss_distill, nan=0.0, posinf=0.0, neginf=0.0)
        return {'loss_distill': loss_distill}

    def loss_edge_consistency(self, outputs, targets, indices, num_boxes, **kwargs):
        loss = outputs.get('loss_edge_consistency')
        if loss is None:
            ref = outputs.get('pred_logits')
            device = ref.device if torch.is_tensor(ref) else torch.device('cpu')
            loss = torch.zeros((), device=device)
        return {'loss_edge_consistency': torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)}



    def _get_distillation_weight_for_epoch(self) -> float:
        fixed_weight = self.weight_dict.get('loss_distill', 0.0)
        return fixed_weight

    def _apply_positive_class_weights(self, loss, idx, target_classes_o):
        """Apply optional per-class weights only on matched positive classes."""
        if (
            self.class_loss_weight_tensor is None
            or target_classes_o.numel() == 0
            or torch.all(self.class_loss_weight_tensor == 1)
        ):
            return loss

        weights = self.class_loss_weight_tensor.to(device=loss.device, dtype=loss.dtype)
        positive_weights = torch.ones_like(loss)
        positive_weights[idx[0], idx[1], target_classes_o] = weights[target_classes_o]
        return loss * positive_weights

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = self._apply_positive_class_weights(loss, idx, target_classes_o)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = self._apply_positive_class_weights(loss, idx, target_classes_o)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = self._apply_positive_class_weights(loss, idx, target_classes_o)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou( \
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        if 'loss_nwd' in self.weight_dict:
            nwd = self.normalized_gaussian_wasserstein_similarity(src_boxes, target_boxes)
            nwd = torch.nan_to_num(nwd, nan=0.0, posinf=0.0, neginf=0.0)
            losses['loss_nwd'] = (1.0 - nwd).sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                    self.fgl_targets_dn = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                    self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                     self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou( \
                box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max + 1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max + 1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                                                                          (F.log_softmax(pred_corners / T, dim=1),
                                                                           F.softmax(target_corners.detach() / T,
                                                                                     dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, (
                                    (~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                                self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'distill': self.loss_distillation,  # NEW: Add distillation loss
            'edge_consistency': self.loss_edge_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss_name in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            if loss_name == 'distill':
                l_dict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                if 'loss_distill' in l_dict:
                    dynamic_weight = self._get_distillation_weight_for_epoch()
                    l_dict['loss_distill'] = l_dict['loss_distill'] * dynamic_weight
                losses.update(l_dict)
            elif loss_name == 'edge_consistency':
                l_dict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)
            else:
                union_losses = ['boxes', 'local'] + (['mal', 'vfl', 'focal'] if self.hdps_class_union else [])
                use_uni_set = self.use_uni_set and (loss_name in union_losses)
                indices_in = indices_go if use_uni_set else indices
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss_name, outputs, targets, indices_in)
                l_dict = self.get_loss(loss_name, outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in ('distill', 'edge_consistency'):
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    union_losses = ['boxes', 'local'] + (['mal', 'vfl', 'focal'] if self.hdps_class_union else [])
                    use_uni_set = self.use_uni_set and (loss in union_losses)
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                if loss in ('distill', 'edge_consistency'):
                    continue
                # TODO, indices and num_box are different from RT-DETRv2
                union_losses = ['boxes', 'local'] + (['mal', 'vfl', 'focal'] if self.hdps_class_union else [])
                use_uni_set = self.use_uni_set and (loss in union_losses)
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    if loss in ('distill', 'edge_consistency'):
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    union_losses = ['boxes'] + (['mal', 'vfl', 'focal'] if self.hdps_class_union else [])
                    use_uni_set = self.use_uni_set and (loss in union_losses)
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in ('distill', 'edge_consistency'):
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss in ('distill', 'edge_consistency'):
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou( \
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum',
                                         avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
               + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list


# Exact source excerpt from engine/solver/det_solver.py:23-387
# Generated for module audit. Do not edit here; inspect original source for execution.

class DetSolver(BaseSolver):
    @staticmethod
    def _per_class_ap50_from_coco(coco_evaluator, hard_class_names=None):
        if coco_evaluator is None or "bbox" not in getattr(coco_evaluator, "coco_eval", {}):
            return {}, None
        coco_eval = coco_evaluator.coco_eval["bbox"]
        if coco_eval.eval is None or "precision" not in coco_eval.eval:
            return {}, None

        precision = coco_eval.eval["precision"]
        params = coco_eval.params
        try:
            iou_idx = int(min(range(len(params.iouThrs)), key=lambda i: abs(float(params.iouThrs[i]) - 0.5)))
            area_idx = list(params.areaRngLbl).index("all") if "all" in params.areaRngLbl else 0
            maxdet_idx = len(params.maxDets) - 1
        except Exception:
            return {}, None

        cat_id_to_name = {
            int(cat_id): coco_eval.cocoGt.cats[int(cat_id)].get("name", str(cat_id))
            for cat_id in params.catIds
            if int(cat_id) in coco_eval.cocoGt.cats
        }
        per_class = {}
        for k, cat_id in enumerate(params.catIds):
            values = precision[iou_idx, :, k, area_idx, maxdet_idx]
            values = values[values > -1]
            if values.size:
                per_class[cat_id_to_name.get(int(cat_id), str(cat_id))] = float(values.mean())

        hard_names = list(hard_class_names or [])
        hard_values = [per_class[name] for name in hard_names if name in per_class]
        hard_mean = float(sum(hard_values) / len(hard_values)) if hard_values else None
        return per_class, hard_mean

    @staticmethod
    def _is_better_v2_1(current_ap50, current_hard, best_ap50, best_hard, tie_threshold):
        if best_ap50 is None:
            return True
        if current_ap50 > best_ap50 + tie_threshold:
            return True
        if abs(current_ap50 - best_ap50) <= tie_threshold:
            return (current_hard if current_hard is not None else -1.0) > (best_hard if best_hard is not None else -1.0)
        return False

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        best_v2_1 = {'epoch': -1, 'ap50': None, 'hard_class_mean_ap50': None}
        best_select_metric = getattr(args, 'best_select_metric', None)
        hard_class_names = getattr(args, 'hard_class_names', [])
        hard_tie_threshold = float(getattr(args, 'hard_class_tie_threshold', 0.003))
        best_ap50_checkpoint_name = getattr(args, 'best_ap50_checkpoint_name', None) or 'best_v2_1.pth'
        best_hardclass_checkpoint_name = getattr(args, 'best_hardclass_checkpoint_name', None)
        best_meta_name = getattr(args, 'best_meta_name', None) or 'best_v2_1.json'
        early_stopping_patience = getattr(args, 'early_stopping_patience', None)
        early_stopping_patience = int(early_stopping_patience) if early_stopping_patience else None
        epochs_since_metric_best = 0
        should_stop_early = False
        best_hardclass = {'epoch': -1, 'hard_class_mean_ap50': None, 'ap50': None}
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
                accumulation_steps=getattr(args, 'accumulation_steps', 1),
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch = epoch
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                elif 'rho' in params and 'delta' in params:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'
                else:
                    reason = 'foreground_params_only'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            per_class_ap50, hard_class_mean_ap50 = self._per_class_ap50_from_coco(coco_evaluator, hard_class_names)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            metric_improved = False
            if (
                best_select_metric in ('ap50_hard_tie', 'ap50_hard_tie_v2_2')
                and 'coco_eval_bbox' in test_stats
                and len(test_stats['coco_eval_bbox']) > 1
            ):
                current_ap50 = float(test_stats['coco_eval_bbox'][1])
                current_map = float(test_stats['coco_eval_bbox'][0])
                current_ap75 = float(test_stats['coco_eval_bbox'][2])
                if self._is_better_v2_1(
                    current_ap50,
                    hard_class_mean_ap50,
                    best_v2_1['ap50'],
                    best_v2_1['hard_class_mean_ap50'],
                    hard_tie_threshold,
                ):
                    best_v2_1 = {
                        'epoch': epoch,
                        'mAP': current_map,
                        'ap50': current_ap50,
                        'ap75': current_ap75,
                        'hard_class_mean_ap50': hard_class_mean_ap50,
                        'per_class_ap50': per_class_ap50,
                        'coco_eval_bbox': test_stats['coco_eval_bbox'],
                    }
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / best_ap50_checkpoint_name)
                        if dist_utils.is_main_process():
                            with (self.output_dir / best_meta_name).open('w') as f:
                                json.dump(best_v2_1, f, indent=2, ensure_ascii=False)
                    metric_improved = True
                    print(f"best_custom_ap50: {best_v2_1}")

                if (
                    best_hardclass_checkpoint_name
                    and hard_class_mean_ap50 is not None
                    and (
                        best_hardclass['hard_class_mean_ap50'] is None
                        or hard_class_mean_ap50 > best_hardclass['hard_class_mean_ap50']
                    )
                ):
                    best_hardclass = {
                        'epoch': epoch,
                        'mAP': current_map,
                        'ap50': current_ap50,
                        'ap75': current_ap75,
                        'hard_class_mean_ap50': hard_class_mean_ap50,
                        'per_class_ap50': per_class_ap50,
                        'coco_eval_bbox': test_stats['coco_eval_bbox'],
                    }
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / best_hardclass_checkpoint_name)
                        if dist_utils.is_main_process():
                            hard_meta_name = best_hardclass_checkpoint_name.rsplit('.', 1)[0] + '.json'
                            with (self.output_dir / hard_meta_name).open('w') as f:
                                json.dump(best_hardclass, f, indent=2, ensure_ascii=False)
                    metric_improved = True
                    print(f"best_hardclass: {best_hardclass}")

            if early_stopping_patience is not None:
                epochs_since_metric_best = 0 if metric_improved else epochs_since_metric_best + 1
                if epochs_since_metric_best >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}: no custom metric improvement "
                        f"for {epochs_since_metric_best} epochs."
                    )
                    should_stop_early = True

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'test_per_class_ap50': per_class_ap50,
                'test_hard_class_mean_ap50': hard_class_mean_ap50,
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if should_stop_early:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
