"""SpectraGate frequency-domain fusion modules (LMSG, CPSE, CCSF) + FSCL.

Original modules developed for the SpectraGate paper, built on the official
Ultralytics YOLO11 framework. Naming follows the FreDFT series (LFEM/NLFEM/FDFTM)
for continuity with the frequency-domain literature; the band-gated cascade
LMSG (LFEM_D_V5) -> CPSE (NLFEM_D_V5) -> CCSF (FDFTM_V6) is our architecture.

Cascade pipeline (P3/P4): LFEM_D_V5 -> NLFEM_D_V5 -> FDFTM_V6
Cascade pipeline (P5)    : LFEM_D_V5 -> FDFTM_V6
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_3d(x):
    """(B, C, H, W) -> (B, H*W, C)"""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1)


def _to_4d(x, h, w):
    """(B, H*W, C) -> (B, C, H, W)"""
    B, _, C = x.shape
    return x.permute(0, 2, 1).reshape(B, C, h, w)


def _channel_shuffle(x, groups):
    """Channel shuffle operation."""
    B, C, H, W = x.size()
    cpg = C // groups
    x = x.view(B, groups, cpg, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, H, W)


# ---------------------------------------------------------------------------
# LayerNorm variants (for FDFTM)
# ---------------------------------------------------------------------------

class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class _LayerNorm(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.body = _WithBiasLayerNorm(dim) if bias else _BiasFreeLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# DeformConv2d (for LFEM)
# ---------------------------------------------------------------------------

class _CBS(nn.Module):
    """Conv-BN-SiLU block."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ===========================================================================
# LFEM - Local Feature Enhancement Module
# ===========================================================================

class _FDFFN(nn.Module):
    """Frequency Domain Feed-Forward Network."""

    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden, hidden, 7, 1, 3, groups=hidden, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden, bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden, hidden, 7, 1, 3, groups=hidden, bias=bias)
        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x3 = self.relu3(self.dwconv3x3(x))
        x5 = self.relu5(self.dwconv5x5(x))
        x7 = self.relu7(self.dwconv7x7(x))

        input_dtype = x3.dtype
        x3_fft = torch.fft.rfft2(x3.float())
        x5_fft = torch.fft.rfft2(x5.float())
        x7_fft = torch.fft.rfft2(x7.float())

        x1_3, x2_3, x3_3 = x3_fft.chunk(3, dim=1)
        x1_5, x2_5, x3_5 = x5_fft.chunk(3, dim=1)
        x1_7, x2_7, x3_7 = x7_fft.chunk(3, dim=1)

        x3_fft = torch.cat([x1_3, x1_5, x1_7], dim=1)
        x5_fft = torch.cat([x2_3, x2_5, x2_7], dim=1)
        x7_fft = torch.cat([x3_3, x3_5, x3_7], dim=1)

        x3 = torch.fft.irfft2(x3_fft, s=(x3.shape[2], x3.shape[3])).to(input_dtype)
        x5 = torch.fft.irfft2(x5_fft, s=(x5.shape[2], x5.shape[3])).to(input_dtype)
        x7 = torch.fft.irfft2(x7_fft, s=(x7.shape[2], x7.shape[3])).to(input_dtype)

        x3 = self.relu3_1(self.dwconv3x3_1(x3))
        x5 = self.relu5_1(self.dwconv5x5_1(x5))
        x7 = self.relu7_1(self.dwconv7x7_1(x7))

        return self.project_out(torch.cat([x3, x5, x7], dim=1))


# ===========================================================================
# FDFTM - Frequency Domain Feature Transform Module
# ===========================================================================

# V5/V6 Experimental Cascade - non-breaking additions
# ===========================================================================


class _FusionRegularizationMixin:
    """Store lightweight differentiable regularization scalars for the current forward."""

    def _init_regularization_state(self):
        self._fusion_regularization = {}

    def __getstate__(self):
        # Exclude transient forward caches so EMA deepcopy does not capture non-leaf tensors.
        state = self.__dict__.copy()
        state["_fusion_regularization"] = {}
        return state

    def _set_regularization_terms(self, **terms):
        self._fusion_regularization = {k: v for k, v in terms.items() if isinstance(v, torch.Tensor)}

    def get_fusion_regularization(self, reset=True):
        terms = getattr(self, "_fusion_regularization", {})
        if reset:
            self._fusion_regularization = {}
        return terms


class _MBSG_V2(_FusionRegularizationMixin, nn.Module):
    """Stabilized multi-band spectral gating with log-ratio modulation."""

    def __init__(self, dim, reduction=4, gate_margin=0.08, cutoff_bounds=(0.15, 0.85),
                 fixed_cutoff=None):
        nn.Module.__init__(self)
        self._init_regularization_state()

        mid = max(dim // reduction, 8)
        self.gate_gen_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, dim * 2, bias=False),
        )
        self.gate_gen_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, dim * 2, bias=False),
        )
        self.band_cutoff = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        self.gate_margin = gate_margin
        self.cutoff_bounds = cutoff_bounds
        self.fixed_cutoff = fixed_cutoff
        if getattr(self, 'fixed_cutoff', None) is not None:
            self.band_cutoff.requires_grad_(False)

    def _get_band_masks(self, H, W, device):
        freq_h = torch.fft.fftfreq(H, device=device).unsqueeze(1).abs()
        freq_w = torch.fft.rfftfreq(W, device=device).unsqueeze(0).abs()
        freq_mag = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        freq_mag = freq_mag / (freq_mag.max() + 1e-8)
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff = self.fixed_cutoff
        else:
            cutoff = self.band_cutoff.sigmoid()
        low_mask = torch.sigmoid(-10.0 * (freq_mag - cutoff))
        return low_mask, 1.0 - low_mask

    def forward(self, primary, auxiliary):
        B, C, H, W = primary.shape
        dtype = primary.dtype

        params_low = self.gate_gen_low(primary).view(B, 2, C, 1, 1).float()
        tau_low, bias_low = params_low[:, 0], params_low[:, 1]
        params_high = self.gate_gen_high(primary).view(B, 2, C, 1, 1).float()
        tau_high, bias_high = params_high[:, 0], params_high[:, 1]

        p_fft = torch.fft.rfft2(primary.float())
        a_fft = torch.fft.rfft2(auxiliary.float())
        p_amp = p_fft.abs()
        a_amp = a_fft.abs()
        ratio = torch.log(a_amp + 1e-6) - torch.log(p_amp + 1e-6)

        gate_low = torch.sigmoid(tau_low * ratio + bias_low)
        gate_high = torch.sigmoid(tau_high * ratio + bias_high)

        low_mask, high_mask = self._get_band_masks(H, W, primary.device)
        low_mask_4d = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask_4d = high_mask.unsqueeze(0).unsqueeze(0)

        gated_fft = p_fft * (gate_low * low_mask_4d + gate_high * high_mask_4d)
        out = torch.fft.irfft2(gated_fft, s=(H, W)).to(dtype)

        energy = gated_fft.abs().pow(2)
        low_energy = (energy * low_mask_4d).mean()
        high_energy = (energy * high_mask_4d).mean()
        total_energy = low_energy + high_energy + 1e-6
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff_reg = torch.zeros((), device=primary.device)
        else:
            cutoff = self.band_cutoff.sigmoid()
            cutoff_low, cutoff_high = self.cutoff_bounds
            cutoff_reg = F.relu(cutoff_low - cutoff) + F.relu(cutoff - cutoff_high)

        self._set_regularization_terms(
            spectral=F.relu(self.gate_margin - low_ratio) + F.relu(self.gate_margin - high_ratio),
            gate=F.relu(self.gate_margin - gate_low.mean()) + F.relu(self.gate_margin - gate_high.mean()),
            cutoff=cutoff_reg,
        )
        return self.act(self.norm(out))


class LFEM_D_V5(_FusionRegularizationMixin, nn.Module):
    """LFEM with stabilized multi-band spectral gating and aux-loss metadata."""

    def __init__(self, in_channels, fixed_cutoff=None):
        nn.Module.__init__(self)
        self._init_regularization_state()
        c = in_channels
        self.CBSk1 = _CBS(c, c, 1, 1)
        self.CBSk3 = _CBS(c, c, 3, 1)
        self.dconv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.mbsg = _MBSG_V2(c, fixed_cutoff=fixed_cutoff)
        self.dwconv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.CBS4C = _CBS(4 * c, c, 1, 1)

    def _enhance(self, primary, auxiliary):
        fea0 = self.CBSk1(primary)
        fea1 = self.CBSk3(fea0)
        fea2 = self.dconv(fea0)
        fea3 = self.mbsg(fea0, auxiliary)
        aux_terms = self.mbsg.get_fusion_regularization(reset=True)
        fea4 = self.dwconv(fea0)
        fea_cat = torch.cat([fea1, fea2, fea3, fea4], dim=1)
        fea_cat = _channel_shuffle(fea_cat, min(32, fea_cat.size(1)))
        return self.CBS4C(fea_cat) + primary, aux_terms

    def forward(self, x):
        rgb_fea, ir_fea = (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, x)
        out_rgb, reg_rgb = self._enhance(rgb_fea, ir_fea)
        out_ir, reg_ir = self._enhance(ir_fea, rgb_fea)
        self._fusion_regularization = {
            key: 0.5 * (reg_rgb[key] + reg_ir[key])
            for key in reg_rgb.keys() & reg_ir.keys()
        }
        return [out_rgb, out_ir]

    def get_fusion_regularization(self, reset=True):
        terms = self._fusion_regularization
        if reset:
            self._fusion_regularization = {}
        return terms


class NLFEM_D_V5(_FusionRegularizationMixin, nn.Module):
    """Cross-spectral enhancement with channel-aware band mixing and signed gating."""

    def __init__(self, in_channels, gate_margin=0.08, cutoff_bounds=(0.15, 0.85),
                 fixed_cutoff=None):
        nn.Module.__init__(self)
        self._init_regularization_state()
        c = in_channels
        mid = max(c // 4, 8)

        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.proj_dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.channel_attn_low = nn.Sequential(
            nn.Linear(c * 2, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, c, bias=False),
        )
        self.channel_attn_high = nn.Sequential(
            nn.Linear(c * 2, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, c, bias=False),
        )
        self.band_mix = nn.Sequential(
            nn.Linear(c * 4, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, 2, bias=False),
        )
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Tanh(),
        )

        self.band_cutoff = nn.Parameter(torch.tensor(0.0))
        self.gate_margin = gate_margin
        self.cutoff_bounds = cutoff_bounds
        self.fixed_cutoff = fixed_cutoff
        if getattr(self, 'fixed_cutoff', None) is not None:
            self.band_cutoff.requires_grad_(False)

    def _get_band_masks(self, H, W, device):
        freq_h = torch.fft.fftfreq(H, device=device).unsqueeze(1).abs()
        freq_w = torch.fft.rfftfreq(W, device=device).unsqueeze(0).abs()
        freq_mag = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        freq_mag = freq_mag / (freq_mag.max() + 1e-8)
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff = self.fixed_cutoff
        else:
            cutoff = self.band_cutoff.sigmoid()
        low = torch.sigmoid(-10.0 * (freq_mag - cutoff))
        return low, 1.0 - low

    def _cross_spectral_enhance(self, primary, auxiliary):
        B, C, H, W = primary.shape
        dtype = primary.dtype

        p = self.proj_dw(self.proj(primary))
        a = self.proj_dw(self.proj(auxiliary))
        p_fft = torch.fft.rfft2(p.float())
        a_fft = torch.fft.rfft2(a.float())

        cross_spec = p_fft * a_fft.conj()
        low_mask, high_mask = self._get_band_masks(H, W, primary.device)
        low_mask_4d = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask_4d = high_mask.unsqueeze(0).unsqueeze(0)

        cross_low = cross_spec * low_mask_4d
        cross_high = cross_spec * high_mask_4d
        cross_low_stat = cross_low.abs().mean(dim=(-2, -1))
        p_low_stat = (p_fft * low_mask_4d).abs().mean(dim=(-2, -1))
        cross_high_stat = cross_high.abs().mean(dim=(-2, -1))
        p_high_stat = (p_fft * high_mask_4d).abs().mean(dim=(-2, -1))

        w_low = self.channel_attn_low(torch.cat([cross_low_stat, p_low_stat], dim=1)).view(B, C, 1, 1).sigmoid()
        w_high = self.channel_attn_high(torch.cat([cross_high_stat, p_high_stat], dim=1)).view(B, C, 1, 1).sigmoid()

        band_ctx = torch.cat([cross_low_stat, p_low_stat, cross_high_stat, p_high_stat], dim=1)
        band_weights = self.band_mix(band_ctx).softmax(dim=1)
        w_band_low = band_weights[:, 0:1].view(B, 1, 1, 1)
        w_band_high = band_weights[:, 1:2].view(B, 1, 1, 1)

        weighted = cross_low * w_low * w_band_low + cross_high * w_high * w_band_high
        spatial_mod = torch.fft.irfft2(weighted, s=(H, W)).to(dtype)
        gate = self.spatial_proj(spatial_mod)

        low_energy = (weighted.abs().pow(2) * low_mask_4d).mean()
        high_energy = (weighted.abs().pow(2) * high_mask_4d).mean()
        total_energy = low_energy + high_energy + 1e-6
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff_reg = torch.zeros((), device=primary.device)
        else:
            cutoff = self.band_cutoff.sigmoid()
            cutoff_low, cutoff_high = self.cutoff_bounds
            cutoff_reg = F.relu(cutoff_low - cutoff) + F.relu(cutoff - cutoff_high)
        reg = {
            "spectral": F.relu(self.gate_margin - low_energy / total_energy) + F.relu(self.gate_margin - high_energy / total_energy),
            "gate": F.relu(self.gate_margin - w_band_low.mean()) + F.relu(self.gate_margin - w_band_high.mean()) + 0.1 * gate.abs().mean(),
            "cutoff": cutoff_reg,
        }
        return primary * (1.0 + gate), reg

    def forward(self, x):
        rgb_fea, ir_fea = (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, x)
        out_rgb, reg_rgb = self._cross_spectral_enhance(rgb_fea, ir_fea)
        out_ir, reg_ir = self._cross_spectral_enhance(ir_fea, rgb_fea)
        self._fusion_regularization = {
            key: 0.5 * (reg_rgb[key] + reg_ir[key])
            for key in reg_rgb.keys() & reg_ir.keys()
        }
        return [out_rgb, out_ir]

    def get_fusion_regularization(self, reset=True):
        terms = self._fusion_regularization
        if reset:
            self._fusion_regularization = {}
        return terms


class _CMFCA_V4(_FusionRegularizationMixin, nn.Module):
    """Conjugate cross-spectral attention with per-band gate summaries."""

    def __init__(self, dim, bias=False, gate_margin=0.08, cutoff_bounds=(0.15, 0.85),
                 fixed_cutoff=None):
        nn.Module.__init__(self)
        self._init_regularization_state()
        self.norm1 = _LayerNorm(dim, bias=True)
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = _LayerNorm(dim * 2, bias=True)

        c_qkv = dim * 2
        mid = max(c_qkv // 4, 8)
        self.freq_gate_low_rgb = nn.Sequential(nn.Linear(c_qkv * 2, mid, bias=False), nn.SiLU(), nn.Linear(mid, c_qkv, bias=False))
        self.freq_gate_low_ir = nn.Sequential(nn.Linear(c_qkv * 2, mid, bias=False), nn.SiLU(), nn.Linear(mid, c_qkv, bias=False))
        self.freq_gate_high_rgb = nn.Sequential(nn.Linear(c_qkv * 2, mid, bias=False), nn.SiLU(), nn.Linear(mid, c_qkv, bias=False))
        self.freq_gate_high_ir = nn.Sequential(nn.Linear(c_qkv * 2, mid, bias=False), nn.SiLU(), nn.Linear(mid, c_qkv, bias=False))

        self.band_cutoff = nn.Parameter(torch.tensor(0.0))
        self.gate_margin = gate_margin
        self.cutoff_bounds = cutoff_bounds
        self.fixed_cutoff = fixed_cutoff
        if getattr(self, 'fixed_cutoff', None) is not None:
            self.band_cutoff.requires_grad_(False)

    def _get_band_masks(self, H, W, device):
        freq_h = torch.fft.fftfreq(H, device=device).unsqueeze(1).abs()
        freq_w = torch.fft.rfftfreq(W, device=device).unsqueeze(0).abs()
        freq_mag = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        freq_mag = freq_mag / (freq_mag.max() + 1e-8)
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff = self.fixed_cutoff
        else:
            cutoff = self.band_cutoff.sigmoid()
        low = torch.sigmoid(-10.0 * (freq_mag - cutoff))
        return low, 1.0 - low

    def _band_gate(self, cross_fft, q_fft, low_mask, high_mask, gate_low_fn, gate_high_fn, B):
        cross_low_amp = (cross_fft * low_mask).abs().mean(dim=(-2, -1))
        q_low_amp = (q_fft * low_mask).abs().mean(dim=(-2, -1))
        g_low = gate_low_fn(torch.cat([cross_low_amp, q_low_amp], dim=1)).view(B, -1, 1, 1).sigmoid()

        cross_high_amp = (cross_fft * high_mask).abs().mean(dim=(-2, -1))
        q_high_amp = (q_fft * high_mask).abs().mean(dim=(-2, -1))
        g_high = gate_high_fn(torch.cat([cross_high_amp, q_high_amp], dim=1)).view(B, -1, 1, 1).sigmoid()

        return cross_fft * low_mask * g_low + cross_fft * high_mask * g_high, g_low.mean(), g_high.mean()

    def forward(self, x):
        rgb_fea, ir_fea = x[0], x[1]
        B, _, H, W = rgb_fea.shape

        rgb_hidden = self.to_hidden_dw(self.to_hidden(self.norm1(rgb_fea)))
        ir_hidden = self.to_hidden_dw(self.to_hidden(self.norm1(ir_fea)))
        rgb_q, rgb_k, rgb_v = rgb_hidden.chunk(3, dim=1)
        ir_q, ir_k, ir_v = ir_hidden.chunk(3, dim=1)

        input_dtype = rgb_q.dtype
        rgb_q_fft = torch.fft.rfft2(rgb_q.float())
        ir_k_fft = torch.fft.rfft2(ir_k.float())
        ir_q_fft = torch.fft.rfft2(ir_q.float())
        rgb_k_fft = torch.fft.rfft2(rgb_k.float())

        rgb_cross = rgb_q_fft * ir_k_fft.conj()
        ir_cross = ir_q_fft * rgb_k_fft.conj()
        low_mask, high_mask = self._get_band_masks(H, W, rgb_fea.device)
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.unsqueeze(0).unsqueeze(0)

        rgb_cross_gated, rgb_low_mean, rgb_high_mean = self._band_gate(
            rgb_cross, rgb_q_fft, low_mask, high_mask, self.freq_gate_low_rgb, self.freq_gate_high_rgb, B
        )
        ir_cross_gated, ir_low_mean, ir_high_mean = self._band_gate(
            ir_cross, ir_q_fft, low_mask, high_mask, self.freq_gate_low_ir, self.freq_gate_high_ir, B
        )

        rgb_out = torch.fft.irfft2(rgb_cross_gated, s=(H, W)).to(input_dtype)
        ir_out = torch.fft.irfft2(ir_cross_gated, s=(H, W)).to(input_dtype)
        rgb_out = self.norm(rgb_out)
        ir_out = self.norm(ir_out)
        rgb_attn = rgb_out.sigmoid()
        ir_attn = ir_out.sigmoid()

        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff_reg = torch.zeros((), device=rgb_fea.device)
        else:
            cutoff = self.band_cutoff.sigmoid()
            cutoff_low, cutoff_high = self.cutoff_bounds
            cutoff_reg = F.relu(cutoff_low - cutoff) + F.relu(cutoff - cutoff_high)
        self._fusion_regularization = {
            "gate": 0.25 * (
                F.relu(self.gate_margin - rgb_low_mean)
                + F.relu(self.gate_margin - rgb_high_mean)
                + F.relu(self.gate_margin - ir_low_mean)
                + F.relu(self.gate_margin - ir_high_mean)
            ),
            "cutoff": cutoff_reg,
        }

        rgb_output = self.project_out(ir_v * rgb_attn)
        ir_output = self.project_out(rgb_v * ir_attn)
        return rgb_output, ir_output

    def get_fusion_regularization(self, reset=True):
        terms = self._fusion_regularization
        if reset:
            self._fusion_regularization = {}
        return terms


class FDFTM_V6(_FusionRegularizationMixin, nn.Module):
    """Conjugate cross-spectral fusion with SiLU output and aux-loss metadata."""

    def __init__(self, dim, gate_margin=0.08, fixed_cutoff=None):
        nn.Module.__init__(self)
        self._init_regularization_state()
        bias = False
        self.attn = _CMFCA_V4(dim, bias=bias, gate_margin=gate_margin, fixed_cutoff=fixed_cutoff)
        self.norm2 = _LayerNorm(dim, bias=True)
        self.ffn = _FDFFN(dim, ffn_expansion_factor=3, bias=bias)
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.act = nn.SiLU()
        self.gate_margin = gate_margin
        self.fixed_cutoff = fixed_cutoff

    def _output_spectral_regularization(self, feat):
        _, _, h, w = feat.shape
        freq_h = torch.fft.fftfreq(h, device=feat.device).unsqueeze(1).abs()
        freq_w = torch.fft.rfftfreq(w, device=feat.device).unsqueeze(0).abs()
        freq_mag = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        freq_mag = freq_mag / (freq_mag.max() + 1e-8)
        if getattr(self, 'fixed_cutoff', None) is not None:
            cutoff = self.fixed_cutoff
        else:
            cutoff = self.attn.band_cutoff.sigmoid()
        low_mask = torch.sigmoid(-10.0 * (freq_mag - cutoff)).unsqueeze(0).unsqueeze(0)
        high_mask = 1.0 - low_mask

        energy = torch.fft.rfft2(feat.float()).abs().pow(2)
        low_energy = (energy * low_mask).mean()
        high_energy = (energy * high_mask).mean()
        total_energy = low_energy + high_energy + 1e-6
        return F.relu(self.gate_margin - low_energy / total_energy) + F.relu(self.gate_margin - high_energy / total_energy)

    def forward(self, x):
        rgb_fea, ir_fea = (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, x)
        rgb_fea_out, ir_fea_out = self.attn([rgb_fea, ir_fea])
        rgb_att_out = rgb_fea + rgb_fea_out
        ir_att_out = ir_fea + ir_fea_out

        rgb_fea = rgb_att_out + self.ffn(self.norm2(rgb_att_out))
        ir_fea = ir_att_out + self.ffn(self.norm2(ir_att_out))
        fused = self.act(self.conv(torch.cat([rgb_fea, ir_fea], dim=1)))

        attn_terms = self.attn.get_fusion_regularization(reset=True)
        self._fusion_regularization = {
            "spectral": self._output_spectral_regularization(fused),
            "gate": attn_terms.get("gate", fused.new_zeros(())),
            "cutoff": attn_terms.get("cutoff", fused.new_zeros(())),
        }
        return fused

    def get_fusion_regularization(self, reset=True):
        terms = self._fusion_regularization
        if reset:
            self._fusion_regularization = {}
        return terms