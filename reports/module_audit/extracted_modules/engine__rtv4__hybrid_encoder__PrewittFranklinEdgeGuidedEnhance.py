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
