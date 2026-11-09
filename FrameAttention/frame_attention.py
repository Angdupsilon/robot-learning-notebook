"""
Frame Attention, experiment version: all heads at once, one attention call per layer.
For a short readable version of the same idea, see frame_gpt.py.

Each head's channels are split into four groups:

    [ free | rope | dials | gyros ]

    free   never rotated, so separate objects can still exchange information
    rope   rotated by fixed position angles (ordinary RoPE); identical in every variant
    dials  2D rotations, inferred from content and updated every layer   -> tags
    gyros  3D rotations, inferred from content and updated every layer   -> structure

With g_i = token i's frame and T_ij = g_i^T g_j ("how j looks from where i stands"):

    score_ij = <q_i, T_ij k_j> / sqrt(d)  +  lam * <f_i, f_j>        f = motion fingerprint (optional)
    out_i    = sum_j A_ij T_ij v_j
    g_i     <- g_i exp( sync * skew( sum_j (A_ij - repel * u_ij) T_ij )   u = uniform weights
                      + move * sum_j A_ij T_ij Omega_j T_ij^T )            Omega_j = turn sent by j

Because T_ij = g_i^T g_j, every pairwise term factors: rotate each token into a shared
"world" frame once, run one ordinary attention call with a few extra value channels,
rotate the results back.
"""

import math
from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- rotations

def rot2(x, angle):
    """Turn each pair in x (..., n, 2) by its angle in angle (..., n)."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1]), dim=-1)


# The 3x3 products below are written out entry by entry on purpose. matmul and einsum on huge
# batches of tiny matrices spend their time on overhead on the GPU; these are plain elementwise
# ops, which torch.compile can merge.

def rot3(x, R, transpose=False):
    """Turn each triple in x (..., n, 3) by its matrix in R (..., n, 3, 3), or by R^T."""
    if transpose:
        R = R.transpose(-1, -2)
    x0, x1, x2 = x.unbind(-1)
    return torch.stack((R[..., 0, 0] * x0 + R[..., 0, 1] * x1 + R[..., 0, 2] * x2,
                        R[..., 1, 0] * x0 + R[..., 1, 1] * x1 + R[..., 1, 2] * x2,
                        R[..., 2, 0] * x0 + R[..., 2, 1] * x1 + R[..., 2, 2] * x2), dim=-1)


def mul3(A, B):
    """A @ B for 3x3 matrices (..., 3, 3)."""
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = A.flatten(-2).unbind(-1)
    b00, b01, b02, b10, b11, b12, b20, b21, b22 = B.flatten(-2).unbind(-1)
    return torch.stack((a00 * b00 + a01 * b10 + a02 * b20, a00 * b01 + a01 * b11 + a02 * b21,
                        a00 * b02 + a01 * b12 + a02 * b22, a10 * b00 + a11 * b10 + a12 * b20,
                        a10 * b01 + a11 * b11 + a12 * b21, a10 * b02 + a11 * b12 + a12 * b22,
                        a20 * b00 + a21 * b10 + a22 * b20, a20 * b01 + a21 * b11 + a22 * b21,
                        a20 * b02 + a21 * b12 + a22 * b22), dim=-1).unflatten(-1, (3, 3))


def hat(w):
    """A spin (axis * speed), shape (..., 3), as a skew-symmetric 3x3 matrix."""
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), dim=-1).reshape(*w.shape, 3)


def vee(M):
    """Read the spin back out of a 3x3 matrix (the inverse of hat, on its skew part)."""
    return 0.5 * torch.stack((M[..., 2, 1] - M[..., 1, 2],
                              M[..., 0, 2] - M[..., 2, 0],
                              M[..., 1, 0] - M[..., 0, 1]), dim=-1)


def so3_exp(w):
    """Rodrigues' formula: spin w -> rotation matrix. Uses a Taylor series near zero."""
    th2 = (w * w).sum(-1, keepdim=True).unsqueeze(-1)
    th = th2.clamp_min(1e-12).sqrt()
    small = th2 < 1e-6
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    K = hat(w)
    x, y, z = w.unbind(-1)
    K2 = torch.stack((-(y * y + z * z), x * y, x * z,  # K @ K = w w^T - |w|^2 I
                      x * y, -(x * x + z * z), y * z,
                      x * z, y * z, -(x * x + y * y)), dim=-1).unflatten(-1, (3, 3))
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + a * K + b * K2


def reorthonormalize(R):
    """One Newton-Schulz step: pulls R back onto the rotations after float drift."""
    return 1.5 * R - 0.5 * mul3(mul3(R, R.transpose(-1, -2)), R)


def inv_softplus(y):
    return math.log(math.expm1(y))


# ----------------------------------------------------------------------------- frame state

class Frames(NamedTuple):
    """Per-head frames, carried from layer to layer next to the residual stream (float32)."""
    dials: torch.Tensor  # (B, H, N, n_dial)       angles
    gyros: torch.Tensor  # (B, H, N, n_gyro, 3, 3) rotation matrices


def uniform_read(x, attn_mask):
    """Average of x (B, H, N, C) over the keys each query is allowed to see."""
    if attn_mask is None:
        return x.mean(-2, keepdim=True).expand_as(x)
    w = attn_mask.to(x.dtype)
    return (w / w.sum(-1, keepdim=True).clamp_min(1.0)) @ x


def _no_autocast(device_type):
    if device_type in ("cuda", "cpu"):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


# ----------------------------------------------------------------------------- the layer

class FrameAttention(nn.Module):
    """
    dim, n_heads              usual attention sizes
    n_free, n_rope            channel pairs per head that are never rotated / rotated by position
    n_dial, n_gyro            number of 2D dials and 3D gyros per head
    fingerprint_dim           width of motion fingerprints (0 = no common-fate term)
    repel                     tokens attended to less than repel * uniform push frames apart
    use_sync, use_moves       switch each frame force on or off
    transport_values          rotate values into the reader's frame (out = sum A T v)
    update_frames             False on the last layer, whose frames nobody reads
    """

    def __init__(self, dim, n_heads, n_free, n_rope, n_dial, n_gyro, fingerprint_dim=0, fp_head_dim=16,
                 repel=1.0, sync_init=0.3, move_init=0.1, use_sync=True, use_moves=True,
                 transport_values=True, update_frames=True):
        super().__init__()
        H = n_heads
        self.H, self.nf, self.nr, self.a, self.b = H, n_free, n_rope, n_dial, n_gyro
        self.hd = n_free + 2 * n_rope + 2 * n_dial + 3 * n_gyro
        self.repel, self.use_sync, self.use_moves = repel, use_sync, use_moves
        has_frames = n_dial + n_gyro > 0
        self.transport_values = transport_values and has_frames
        self.update = update_frames and has_frames and (use_sync or use_moves)

        self.qkv = nn.Linear(dim, 3 * H * self.hd, bias=False)
        self.out = nn.Linear(H * self.hd, dim, bias=False)
        n_blocks = n_dial + n_gyro
        if self.update and use_sync:
            self.sync_rate = nn.Parameter(torch.full((H, n_blocks), inv_softplus(sync_init)))
        if self.update and use_moves:
            self.move = nn.Linear(dim, H * (n_dial + 3 * n_gyro), bias=False)  # the turn each token sends
            self.move_rate = nn.Parameter(torch.full((H, n_blocks), inv_softplus(move_init)))
        self.fp_dim = fingerprint_dim
        if fingerprint_dim:
            self.fp = nn.Linear(fingerprint_dim, H * fp_head_dim, bias=False)
            self.lam = nn.Parameter(torch.full((H,), inv_softplus(0.1)))

    def _rotate(self, x, dials, gyros, rope=None, inverse=False):
        """Rotate the dial/gyro channels of x (B, H, N, hd) by each token's frame (or undo it).
        rope (B|1, 1, N, n_rope) angles are applied to the rope channels when given."""
        free, r, d, g = x.split((self.nf, 2 * self.nr, 2 * self.a, 3 * self.b), dim=-1)
        if rope is not None and self.nr:
            r = rot2(r.unflatten(-1, (self.nr, 2)), rope).flatten(-2)
        if self.a:
            d = rot2(d.unflatten(-1, (self.a, 2)), -dials if inverse else dials).flatten(-2)
        if self.b:
            g = rot3(g.unflatten(-1, (self.b, 3)), gyros, transpose=inverse).flatten(-2)
        return torch.cat((free, r, d, g), dim=-1).to(x.dtype)

    def forward(self, h, frames, rope=None, fingerprints=None, attn_mask=None):
        """
        h            (B, N, dim)
        frames       Frames with float32 dials (B, H, N, a) and gyros (B, H, N, b, 3, 3)
        rope         (B|1, 1, N, n_rope) position angles, or None
        fingerprints (B, N, fingerprint_dim) or None
        attn_mask    bool, broadcastable to (B, H, N, N), True = may attend
        returns      (out (B, N, dim), new Frames)
        """
        B, N, _ = h.shape
        H, hd, a, b = self.H, self.hd, self.a, self.b
        dials, gyros = frames
        q, k, v = self.qkv(h).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)

        # 1) move queries, keys (and values) out of each token's own frame into the world frame
        q = self._rotate(q, dials, gyros, rope)
        k = self._rotate(k, dials, gyros, rope)
        if self.transport_values:
            v = self._rotate(v, dials, gyros)
        q = q * hd ** -0.5
        if self.fp_dim and fingerprints is not None:
            f = self.fp(fingerprints).view(B, N, H, -1).transpose(1, 2)
            f = (f * F.softplus(self.lam).sqrt().view(1, H, 1, 1)).to(q.dtype)
            q, k = torch.cat((q, f), -1), torch.cat((k, f), -1)

        # 2) one attention call. Extra value channels carry what the frame update needs:
        #    world-frame dials (cos, sin) and gyros for sync; the turns sent (theta, R w) for moves.
        vals = [v]
        if self.update and self.use_sync:
            vals += [torch.cos(dials), torch.sin(dials), gyros.flatten(-3)]
        if self.update and self.use_moves:
            m = self.move(h).view(B, N, H, a + 3 * b).transpose(1, 2).to(dials.dtype)
            theta, w = m[..., :a], m[..., a:].unflatten(-1, (b, 3))
            vals += [theta, rot3(w, gyros).flatten(-2)]
        vals = torch.cat([x.to(q.dtype) for x in vals], dim=-1)
        read = F.scaled_dot_product_attention(q, k, vals, attn_mask=attn_mask, scale=1.0)

        # 3) values back into each token's own frame
        o = read[..., :hd]
        if self.transport_values:
            o = self._rotate(o, dials, gyros, inverse=True)
        out = self.out(o.transpose(1, 2).reshape(B, N, H * hd))
        if not self.update:
            return out, frames
        return out, self._update_frames(read[..., hd:], dials, gyros, attn_mask, h.device.type)

    def _update_frames(self, read, dials, gyros, attn_mask, device_type):
        H, a, b = self.H, self.a, self.b
        with _no_autocast(device_type):  # frames always update in their own precision (float32)
            read = read.to(dials.dtype)
            dial_step = torch.zeros_like(dials)
            gyro_step = torch.zeros(gyros.shape[:-1], dtype=gyros.dtype, device=gyros.device)
            if self.use_sync:
                rc, rs, rM, read = read.split((a, a, 9 * b, read.shape[-1] - 2 * a - 9 * b), dim=-1)
                base = uniform_read(torch.cat((torch.cos(dials), torch.sin(dials), gyros.flatten(-3)), -1), attn_mask)
                bc, bs, bM = base.split((a, a, 9 * b), dim=-1)
                mc, ms = rc - self.repel * bc, rs - self.repel * bs
                # sum_j (A_ij - k u_ij) sin(phi_j - phi_i), using sin(x - y) = sin x cos y - cos x sin y
                dial_sync = ms * torch.cos(dials) - mc * torch.sin(dials)
                M = (rM - self.repel * bM).unflatten(-1, (b, 3, 3))
                gyro_sync = vee(mul3(gyros.transpose(-1, -2), M))  # same rule for 3D rotations
                eta = F.softplus(self.sync_rate)
                dial_step = dial_step + eta[:, :a].view(1, H, 1, a) * dial_sync
                gyro_step = gyro_step + eta[:, a:].view(1, H, 1, b, 1) * gyro_sync
            if self.use_moves:
                theta_in, u = read.split((a, 3 * b), dim=-1)
                eta = F.softplus(self.move_rate)
                dial_step = dial_step + eta[:, :a].view(1, H, 1, a) * theta_in  # 2D turns just add
                gyro_move = rot3(u.unflatten(-1, (b, 3)), gyros, transpose=True)  # into my frame
                gyro_step = gyro_step + eta[:, a:].view(1, H, 1, b, 1) * gyro_move
            new_dials = dials + dial_step
            new_gyros = reorthonormalize(mul3(gyros, so3_exp(gyro_step)))
        return Frames(new_dials, new_gyros)


# ----------------------------------------------------------------------------- the published baseline

def _rotate_pairs(x, c, s):
    """Turn each (real, imag) pair in the last dim of x (..., 2n) by the angles with cos c and sin s (..., n)."""
    re, im = x.unflatten(-1, (x.shape[-1] // 2, 2)).unbind(-1)
    return torch.stack((re * c - im * s, re * s + im * c), dim=-1).flatten(-2).to(x.dtype)


class KoPEAttention(nn.Module):
    """
    KoPE attention (arXiv 2604.07904), ported from vit_kope.py in
    github.com/microsoft/Neuro-inspired_Phase_Encoding with its default settings.

    Every channel pair of a head has one phase. A learned per-head matrix mixes the phases, then
    q, k and v are rotated by them and the attention output is rotated back. Phases are updated
    between layers by KoPECoupling, not here. Takes and returns Frames so it fits the same Block.
    """

    def __init__(self, dim, n_heads, mix_gain=0.1):
        super().__init__()
        self.H, self.hd = n_heads, dim // n_heads
        assert self.hd % 4 == 0, "KoPE needs a head dim divisible by 4"
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        n = self.hd // 2
        mix = nn.init.xavier_normal_(torch.empty(n_heads, n, n), gain=mix_gain)
        self.mix = nn.Parameter(mix + torch.eye(n))

    def _mixed(self, dials):
        with _no_autocast(dials.device.type):
            mix = self.mix.to(dials.dtype)[None]  # (1, H, n, n)
            c, s = torch.cos(dials) @ mix, torch.sin(dials) @ mix
            inv = (c.square() + s.square()).clamp_min(1e-5).rsqrt()
            return c * inv, s * inv

    def used_frames(self, frames):
        """The phases this layer actually rotates by (after mixing), for the grouping metrics."""
        c, s = self._mixed(frames.dials)
        return Frames(torch.atan2(s, c), frames.gyros)

    def forward(self, x, frames, rope=None, fingerprints=None, attn_mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        c, s = self._mixed(frames.dials)
        q, k, v = _rotate_pairs(q, c, s), _rotate_pairs(k, c, s), _rotate_pairs(v, c, s)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = _rotate_pairs(o, c, -s)
        return self.out(o.transpose(1, 2).reshape(B, N, C)), frames


class KoPECoupling(nn.Module):
    """
    One Kuramoto step on the phases, shared by every layer: a separate q/k attention whose values
    are the phases themselves. KoPE keeps (cos, sin) and does r <- normalize(r + gamma * tangent),
    where tangent is the part of the attention read perpendicular to r. On the unit circle that
    is exactly phi <- phi + atan(gamma * sum_j J_ij sin(phi_j - phi_i)), which is what is coded.
    """

    def __init__(self, dim, n_heads, gamma=0.05):
        super().__init__()
        self.H, self.hd, self.gamma = n_heads, dim // n_heads, gamma
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.qk_scale = nn.Parameter(torch.ones(1, n_heads, 1, self.hd))

    def forward(self, x, frames, attn_mask=None):
        B, N, _ = x.shape
        dials = frames.dials
        q = F.rms_norm(self.q_proj(x).view(B, N, self.H, self.hd).transpose(1, 2), (self.hd,))
        k = F.rms_norm(self.k_proj(x).view(B, N, self.H, self.hd).transpose(1, 2), (self.hd,))
        q = q * self.qk_scale.to(q.dtype)
        cos, sin = torch.cos(dials), torch.sin(dials)
        read = F.scaled_dot_product_attention(q, k, torch.cat((cos, sin), -1).to(q.dtype), attn_mask=attn_mask)
        with _no_autocast(dials.device.type):
            rc, rs = read.to(dials.dtype).chunk(2, dim=-1)
            return Frames(dials + torch.atan(self.gamma * (rs * cos - rc * sin)), frames.gyros)


def identity_frames(B, H, N, n_dial, n_gyro, device=None):
    dials = torch.zeros(B, H, N, n_dial, device=device)
    gyros = torch.eye(3, device=device).expand(B, H, N, n_gyro, 3, 3).clone()
    return Frames(dials, gyros)
