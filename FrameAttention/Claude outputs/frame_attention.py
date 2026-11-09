"""
Frame Attention
===============

One attention layer that packages three binding mechanisms:

    #1 phase tags      -> 2D rotation "dials"       (commute:       who belongs together)
    #2 common fate     -> motion fingerprints       (optional:      who moves together)
    #3 operator tokens -> 3D rotation "gyroscopes"  (don't commute: how things are arranged)

Every token carries content h_i and, per head, a frame g_i in SO(2)^a x SO(3)^b.
With T_ij = g_i^T g_j ("how token j looks from where token i stands"):

    s_ij  = <q_i, T_ij k_j> / sqrt(d)  +  lam * <w_i, w_j>                  score
    A     = softmax(s)
    out_i = sum_j A_ij T_ij v_j                                              read
    g_i  <- g_i exp( eta_s * skew( sum_j (A_ij - kappa u_ij) T_ij F^T )      sync  (Lohe / Kuramoto)
                   + eta_c * sum_j A_ij T_ij Omega_j T_ij^T )                compose

Because T_ij = g_i^T g_j, every pairwise term factors:
rotate each token into a shared world frame once, run ONE standard attention call
with a few extra value channels, rotate the results back. No N^2 matrix products.

Special cases (checked in test_frame_attention.py):
    all frames = identity                         -> standard multi-head attention
    dials = position * freq, transport_values=False -> RoPE, exactly
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def rot2(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Rotate 2-vectors x[..., k, :] by angles phi[..., k]."""
    c, s = torch.cos(phi), torch.sin(phi)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((c * x0 - s * x1, s * x0 + c * x1), dim=-1)


def rot3(x: torch.Tensor, R: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Apply rotations R[..., k, :, :] (or their transposes) to 3-vectors x[..., k, :]."""
    return torch.einsum("...ji,...j->...i" if transpose else "...ij,...j->...i", R, x)


def hat(w: torch.Tensor) -> torch.Tensor:
    """3-vector -> skew-symmetric matrix [w]_x. Plain terms: a spin rate as a matrix."""
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*w.shape, 3)


def vee(M: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric part of a 3x3 matrix -> 3-vector (inverse of `hat`)."""
    return 0.5 * torch.stack(
        (M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]), -1
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: spin rate w (axis * angle) -> rotation matrix. Safe at w = 0."""
    th2 = (w * w).sum(-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    th = th2.clamp_min(1e-12).sqrt()
    small = th2 < 1e-6
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    K = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + a * K + b * (K @ K)


def reorthonormalize(R: torch.Tensor) -> torch.Tensor:
    """One Newton-Schulz step: pulls R back onto SO(3) after floating-point drift."""
    return 1.5 * R - 0.5 * R @ R.transpose(-1, -2) @ R


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


# --------------------------------------------------------------------------- #
# Frame state
# --------------------------------------------------------------------------- #
@dataclass
class Frames:
    """Per-head token frames, carried from layer to layer like a second residual stream.

    dials: (B, H, N, a)        angles of 2D rotations  -> tags (commute)
    gyros: (B, H, N, b, 3, 3)  3D rotation matrices    -> structure (don't commute)
    """

    dials: torch.Tensor
    gyros: torch.Tensor

    def order_parameters(self) -> dict:
        """Synchrony diagnostics in [0, 1]. Values near 1 mean the frames have
        collapsed to one setting and the layer has silently become standard attention."""
        out = {}
        if self.dials.shape[-1]:
            r = torch.complex(torch.cos(self.dials), torch.sin(self.dials)).mean(-2).abs()
            out["dial_sync"] = r.mean().item()
        if self.gyros.shape[-3]:
            out["gyro_sync"] = (self.gyros.mean(-4).flatten(-2).norm(dim=-1) / math.sqrt(3)).mean().item()
        return out


def _uniform_read(x: torch.Tensor, mask: torch.Tensor | None, causal: bool) -> torch.Tensor:
    """Average of x over the keys each query is allowed to see: the u_ij baseline
    for the repulsion term. x: (B, H, N, C)."""
    N = x.shape[-2]
    if mask is None and not causal:
        return x.mean(-2, keepdim=True).expand_as(x)
    if mask is None:
        counts = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).view(N, 1)
        return x.cumsum(-2) / counts
    w = mask.to(x.dtype)
    return (w / w.sum(-1, keepdim=True).clamp_min(1.0)) @ x


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #
class FrameAttention(nn.Module):
    """Frame Attention.

    Head layout: head_dim = n_free + 2 * n_dials + 3 * n_gyros
        n_free   channels are never rotated: the "relations channel" that lets
                 separate objects still talk to each other.
        n_dials  2D blocks rotated by the dial angles.
        n_gyros  3D blocks rotated by the gyroscope matrices.

    Args:
        fingerprint_dim:  width of motion fingerprints passed to forward (0 = no common fate).
        repel:            kappa. Tokens attended to less than kappa * uniform push frames apart.
        eta_sync, eta_comp: initial step sizes; learned per head and per block, so each block
                          decides for itself whether it behaves more like a tag or like structure.
        frustration:      learn a preferred nonzero offset per block (Kuramoto-Sakaguchi).
        transport_values: rotate values into the reader's frame (out = sum A T v).
                          False gives exactly RoPE when dials are set from positions.
        steps:            synchronisation steps per layer (depth also supplies steps).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_free: int = 16,
        n_dials: int = 16,
        n_gyros: int = 8,
        fingerprint_dim: int = 0,
        fp_head_dim: int = 16,
        repel: float = 1.0,
        eta_sync: float = 0.3,
        eta_comp: float = 0.1,
        frustration: bool = True,
        transport_values: bool = True,
        steps: int = 1,
    ):
        super().__init__()
        H = n_heads
        self.H, self.nf, self.a, self.b = H, n_free, n_dials, n_gyros
        self.hd = n_free + 2 * n_dials + 3 * n_gyros
        self.repel, self.steps, self.transport_values = repel, steps, transport_values

        self.qkv = nn.Linear(dim, 3 * H * self.hd, bias=False)
        # Omega_j: each token's "move" (a spin rate per block), sent to whoever attends to it.
        self.spin = nn.Linear(dim, H * (n_dials + 3 * n_gyros), bias=False)
        nn.init.normal_(self.spin.weight, std=1e-3)
        self.out = nn.Linear(H * self.hd, dim, bias=False)

        n_blocks = n_dials + n_gyros
        self.eta_sync = nn.Parameter(torch.full((H, n_blocks), _inv_softplus(eta_sync)))
        self.eta_comp = nn.Parameter(torch.full((H, n_blocks), _inv_softplus(eta_comp)))
        # Frustration F: preferred relative offset ("the handle sits at a fixed angle to the mug").
        self.alpha = nn.Parameter(torch.zeros(H, n_dials), requires_grad=frustration)
        self.f_gyro = nn.Parameter(torch.zeros(H, n_gyros, 3), requires_grad=frustration)

        self.fp_dim = fingerprint_dim
        if fingerprint_dim:
            self.fph = fp_head_dim
            self.fp = nn.Linear(fingerprint_dim, H * fp_head_dim, bias=False)
            self.lam = nn.Parameter(torch.full((H,), _inv_softplus(0.1)))

    # -- rotate head vectors between a token's own frame and the world frame --------- #
    def _rotate(self, x: torch.Tensor, phi: torch.Tensor, R: torch.Tensor, inverse: bool) -> torch.Tensor:
        nf, a, b = self.nf, self.a, self.b
        free, d, g = x.split((nf, 2 * a, 3 * b), -1)
        d = rot2(d.unflatten(-1, (a, 2)), -phi if inverse else phi).flatten(-2)
        g = rot3(g.unflatten(-1, (b, 3)), R, transpose=inverse).flatten(-2)
        return torch.cat((free, d, g), -1)

    def forward(
        self,
        h: torch.Tensor,
        frames: Frames,
        fingerprints: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ):
        """
        h:            (B, N, dim) content
        frames:       Frames for this layer's heads
        fingerprints: (B, N, fingerprint_dim) motion fingerprints, or None
        attn_mask:    bool, broadcastable to (B, H, N, N); True = may attend
        Returns (out, new_frames, info): out is (B, N, dim), to be added to the residual stream.
        """
        B, N, _ = h.shape
        H, a, b, hd = self.H, self.a, self.b, self.hd

        q, k, v = self.qkv(h).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)  # each (B, H, N, hd)
        spin = self.spin(h).view(B, N, H, a + 3 * b).transpose(1, 2)
        theta, w = spin[..., :a], spin[..., a:].unflatten(-1, (b, 3))  # Omega_j for dials / gyros

        if attn_mask is not None and is_causal:
            attn_mask = attn_mask & torch.ones(N, N, dtype=torch.bool, device=h.device).tril()
            is_causal = False

        fpq = None
        if self.fp_dim and fingerprints is not None:
            fpq = self.fp(fingerprints).view(B, N, H, self.fph).transpose(1, 2)
            fpq = fpq * F.softplus(self.lam).sqrt().view(1, H, 1, 1)  # <fpq_i, fpq_j> = lam <w_i, w_j>

        eta_s, eta_c = F.softplus(self.eta_sync), F.softplus(self.eta_comp)  # (H, a + b)
        eds, edc = eta_s[:, :a].view(1, H, 1, a), eta_c[:, :a].view(1, H, 1, a)
        egs, egc = eta_s[:, a:].view(1, H, 1, b, 1), eta_c[:, a:].view(1, H, 1, b, 1)
        Fg = so3_exp(self.f_gyro)  # (H, b, 3, 3)

        phi, R = frames.dials, frames.gyros
        for _ in range(self.steps):
            # 1. Lift into the world frame: x -> g x.  Then <g_i q_i, g_j k_j> = <q_i, T_ij k_j>.
            qw, kw = self._rotate(q, phi, R, False), self._rotate(k, phi, R, False)
            vw = self._rotate(v, phi, R, False) if self.transport_values else v
            qs, ks = qw / math.sqrt(hd), kw
            if fpq is not None:
                qs, ks = torch.cat((qs, fpq), -1), torch.cat((ks, fpq), -1)

            # 2. One standard attention call. Extra value channels carry each key's
            #    world-frame dial (cos, sin), dial spin, gyro matrix, and gyro spin (R_j w_j),
            #    because T_ij Omega_j T_ij^T = R_i^T [R_j w_j]_x R_i.
            vals = torch.cat(
                (vw, torch.cos(phi), torch.sin(phi), theta, R.flatten(-3), rot3(w, R).flatten(-2)), -1
            )
            read = F.scaled_dot_product_attention(qs, ks, vals, attn_mask=attn_mask, is_causal=is_causal, scale=1.0)
            o, mc, ms, th_in, M, u = read.split((hd, a, a, a, 9 * b, 3 * b), -1)

            # 3. Read values back into each token's own frame: out_i = sum_j A_ij T_ij v_j.
            o = self._rotate(o, phi, R, True) if self.transport_values else o

            # 4. Frame update. Repulsion subtracts kappa * (uniform average over visible keys).
            base = _uniform_read(torch.cat((torch.cos(phi), torch.sin(phi), R.flatten(-3)), -1), attn_mask, is_causal)
            bc, bs, bR = base.split((a, a, 9 * b), -1)
            k_ = self.repel

            #    Dials: sum_j (A_ij - k u_ij) sin(phi_j - phi_i - alpha) = Im(e^{-i(phi_i+alpha)} m_i)
            psi = phi + self.alpha.view(1, H, 1, a)
            mc_, ms_ = mc - k_ * bc, ms - k_ * bs
            dial_sync = ms_ * torch.cos(psi) - mc_ * torch.sin(psi)
            phi_new = phi + eds * dial_sync + edc * th_in  # 2D rotations commute: compose = add

            #    Gyros: sync = vee(R_i^T (M_i - k Mbar_i) F^T) ; compose = R_i^T u_i
            Mi = (M - k_ * bR).unflatten(-1, (b, 3, 3))
            gyro_sync = vee(R.transpose(-1, -2) @ Mi @ Fg.transpose(-1, -2).unsqueeze(1))
            gyro_comp = rot3(u.unflatten(-1, (b, 3)), R, transpose=True)
            R_new = reorthonormalize(R @ so3_exp(egs * gyro_sync + egc * gyro_comp))

            phi, R = phi_new, R_new

        out = self.out(o.transpose(1, 2).reshape(B, N, H * hd))
        new = Frames(phi, R)
        return out, new, new.order_parameters()


# --------------------------------------------------------------------------- #
# Initial frames, motion fingerprints, and a residual block
# --------------------------------------------------------------------------- #
class FrameInit(nn.Module):
    """Initial frames from content, plus optional RoPE angles on the dials.
    Zero-initialised, so a fresh model starts as a standard (or RoPE) transformer."""

    def __init__(self, dim: int, n_heads: int, n_dials: int, n_gyros: int, rope_base: float = 10000.0):
        super().__init__()
        self.H, self.a, self.b = n_heads, n_dials, n_gyros
        self.proj = nn.Linear(dim, n_heads * (n_dials + 3 * n_gyros))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("freq", rope_base ** (-torch.arange(n_dials, dtype=torch.float32) / max(n_dials, 1)))

    def forward(self, h: torch.Tensor, positions: torch.Tensor | None = None) -> Frames:
        B, N, _ = h.shape
        z = self.proj(h).view(B, N, self.H, self.a + 3 * self.b).transpose(1, 2)
        phi = z[..., : self.a]
        if positions is not None:  # (N,) or (B, N)
            pos = positions.to(h.dtype).reshape(-1 if positions.dim() == 2 else 1, 1, N, 1)
            phi = phi + pos * self.freq.to(h.dtype)
        return Frames(phi, so3_exp(z[..., self.a :].unflatten(-1, (self.b, 3))))


class CommonFateProbe(nn.Module):
    """Motion fingerprints: poke a learned dynamics model and record how each token responds.

    <fp_i, fp_j> averages delta_i . delta_j over random pokes, a Hutchinson estimate of how
    alike the two tokens' sensitivities are ("do we move together?").
    Train `dyn` with `dynamics_loss` on (h_t, h_{t+1}) pairs from video or robot logs.
    If you already have real motion (optical flow, robot actions), skip this and pass
    those features to FrameAttention as fingerprints directly.
    """

    def __init__(self, dim: int, fp_dim: int = 8, n_pokes: int = 8, poke_scale: float = 0.1,
                 poke_frac: float = 0.25, n_heads: int = 4):
        super().__init__()
        self.S, self.eps, self.frac = n_pokes, poke_scale, poke_frac
        self.dyn = nn.TransformerEncoderLayer(dim, n_heads, 2 * dim, dropout=0.0, batch_first=True)
        self.proj = nn.Linear(dim, fp_dim, bias=False)
        self.out_dim = n_pokes * fp_dim

    def dynamics_loss(self, h_t: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.dyn(h_t), h_next.detach())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        S = self.S
        base = self.dyn(h)
        hit = (torch.rand(B, S, N, 1, device=h.device) < self.frac).to(h.dtype)  # each poke nudges a few tokens
        pokes = torch.randn(B, S, N, D, device=h.device, dtype=h.dtype) * hit * self.eps
        resp = self.dyn((h.unsqueeze(1) + pokes).flatten(0, 1)).view(B, S, N, D)
        delta = (resp - base.unsqueeze(1)) / self.eps  # how each token's prediction moved
        fp = self.proj(delta).permute(0, 2, 1, 3)  # (B, N, S, fp_dim)
        return fp.flatten(-2) / math.sqrt(S)


class FrameBlock(nn.Module):
    """Pre-norm residual block. Content goes through the residual stream as usual;
    frames are carried alongside and updated by the attention layer."""

    def __init__(self, dim: int, mlp_ratio: int = 4, **attn_kwargs):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = FrameAttention(dim, **attn_kwargs)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, h, frames, fingerprints=None, **kw):
        a, frames, info = self.attn(self.norm1(h), frames, fingerprints, **kw)
        h = h + a
        h = h + self.mlp(self.norm2(h))
        return h, frames, info


if __name__ == "__main__":
    torch.manual_seed(0)
    dim, H, a, b = 64, 4, 4, 2
    x = torch.randn(2, 10, dim)
    init = FrameInit(dim, H, a, b)
    probe = CommonFateProbe(dim)
    blocks = nn.ModuleList(
        FrameBlock(dim, n_heads=H, n_free=4, n_dials=a, n_gyros=b, fingerprint_dim=probe.out_dim)
        for _ in range(3)
    )
    frames, fp = init(x, positions=torch.arange(10)), probe(x)
    for blk in blocks:
        x, frames, info = blk(x, frames, fp)
    print("out", tuple(x.shape), "| dials", tuple(frames.dials.shape), "| gyros", tuple(frames.gyros.shape))
    print("order parameters:", {k: round(v, 3) for k, v in info.items()})
