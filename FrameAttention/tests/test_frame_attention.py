"""
Checks that the fast Frame Attention layer does exactly what the math says.

The main test rebuilds every pairwise relative frame T_ij = g_i^T g_j explicitly (the slow
N^2 way) and confirms the fast "rotate once, one attention call" layer gives the same output
and the same frame update.  Run:  python -m pytest tests -q
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame_attention import (FrameAttention, Frames, KoPEAttention, KoPECoupling, hat, identity_frames,  # noqa: E402
                             reorthonormalize, so3_exp, vee)


def layer(dtype=torch.float64, **kw):
    torch.manual_seed(0)
    cfg = dict(dim=24, n_heads=3, n_free=4, n_rope=2, n_dial=2, n_gyro=2, fingerprint_dim=5, fp_head_dim=4)
    cfg.update(kw)
    L = FrameAttention(**cfg).to(dtype)
    with torch.no_grad():  # make every mechanism matter
        if hasattr(L, "move"):
            L.move.weight.normal_(0, 0.3)
        for name in ("sync_rate", "move_rate"):
            if hasattr(L, name):
                getattr(L, name).normal_(0, 0.5)
    return L


def random_frames(B, H, N, a, b, dtype=torch.float64):
    return Frames(torch.randn(B, H, N, a, dtype=dtype) * 2, so3_exp(torch.randn(B, H, N, b, 3, dtype=dtype) * 2))


def naive(L, h, frames, rope, fp, keep=None):
    """Explicit T_ij for every pair. keep: (B, N) bool key mask or None."""
    B, N, _ = h.shape
    H, nf, nr, a, b, hd = L.H, L.nf, L.nr, L.a, L.b, L.hd
    q, k, v = L.qkv(h).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
    dials, gyros = frames
    G = torch.zeros(B, H, N, hd, hd, dtype=h.dtype)  # each token's full frame matrix
    G[..., :nf + 2 * nr, :nf + 2 * nr] = torch.eye(nf + 2 * nr, dtype=h.dtype)  # free and rope: untouched
    for i in range(a):
        c, s = torch.cos(dials[..., i]), torch.sin(dials[..., i])
        j = nf + 2 * nr + 2 * i
        G[..., j, j], G[..., j, j + 1], G[..., j + 1, j], G[..., j + 1, j + 1] = c, -s, s, c
    for i in range(b):
        j = nf + 2 * nr + 2 * a + 3 * i
        G[..., j:j + 3, j:j + 3] = gyros[..., i, :, :]
    Rp = torch.zeros(1, 1, N, hd, hd, dtype=h.dtype)  # rope: only on queries and keys
    Rp[..., :, :] = torch.eye(hd, dtype=h.dtype)
    for i in range(nr):
        c, s = torch.cos(rope[0, 0, :, i]), torch.sin(rope[0, 0, :, i])
        j = nf + 2 * i
        Rp[0, 0, :, j, j], Rp[0, 0, :, j, j + 1], Rp[0, 0, :, j + 1, j], Rp[0, 0, :, j + 1, j + 1] = c, -s, s, c
    T = torch.einsum("bhiyx,bhjyz->bhijxz", G, G)
    TR = torch.einsum("bhiyx,bhjyz->bhijxz", Rp.expand(B, H, N, hd, hd) @ G, Rp.expand(B, H, N, hd, hd) @ G)
    s = torch.einsum("bhix,bhijxz,bhjz->bhij", q, TR, k) / math.sqrt(hd)
    f = L.fp(fp).view(B, N, H, -1).transpose(1, 2)
    s = s + F.softplus(L.lam).view(1, H, 1, 1) * f @ f.transpose(-1, -2)
    allowed = torch.ones(B, 1, N, N, dtype=torch.bool) if keep is None else keep[:, None, None, :].expand(B, 1, N, N)
    A = s.masked_fill(~allowed, float("-inf")).softmax(-1)
    u = allowed.to(h.dtype) / allowed.to(h.dtype).sum(-1, keepdim=True)
    o = torch.einsum("bhij,bhijxz,bhjz->bhix", A, T, v)
    out = L.out(o.transpose(1, 2).reshape(B, N, H * hd))

    m = L.move(h).view(B, N, H, a + 3 * b).transpose(1, 2)
    theta, w = m[..., :a], m[..., a:].unflatten(-1, (b, 3))
    Wr = A - L.repel * u
    es, em = F.softplus(L.sync_rate), F.softplus(L.move_rate)
    dphi = dials.unsqueeze(-3) - dials.unsqueeze(-2)  # [i, j] = phi_j - phi_i
    new_d = (dials + es[:, :a].view(1, H, 1, a) * torch.einsum("bhij,bhija->bhia", Wr, torch.sin(dphi))
             + em[:, :a].view(1, H, 1, a) * torch.einsum("bhij,bhja->bhia", A, theta))
    Tg = torch.einsum("bhigyx,bhjgyz->bhijgxz", gyros, gyros)
    sync = vee(torch.einsum("bhij,bhijgxy->bhigxy", Wr, Tg))
    comp = vee(torch.einsum("bhij,bhijgxy->bhigxy", A, Tg @ hat(w).unsqueeze(2) @ Tg.transpose(-1, -2)))
    xi = es[:, a:].view(1, H, 1, b, 1) * sync + em[:, a:].view(1, H, 1, b, 1) * comp
    return out, Frames(new_d, reorthonormalize(gyros @ so3_exp(xi)))


@pytest.mark.parametrize("masked", [False, True])
def test_fast_equals_naive_pairwise(masked):
    L = layer()
    torch.manual_seed(1)
    B, N = 2, 7
    h, fp = torch.randn(B, N, 24, dtype=torch.float64), torch.randn(B, N, 5, dtype=torch.float64)
    fr, rope = random_frames(B, L.H, N, L.a, L.b), torch.randn(1, 1, N, L.nr, dtype=torch.float64) * 3
    keep = torch.ones(B, N, dtype=torch.bool)
    if masked:
        keep[0, -2:] = False
    out, new = L(h, fr, rope, fp, attn_mask=keep[:, None, None, :] if masked else None)
    out_n, new_n = naive(L, h, fr, rope, fp, keep if masked else None)
    assert torch.allclose(out, out_n, atol=1e-10)
    assert torch.allclose(new.dials, new_n.dials, atol=1e-10)
    assert torch.allclose(new.gyros, new_n.gyros, atol=1e-10)


def test_identity_frames_no_rope_is_standard_attention():
    L = layer(fingerprint_dim=0)
    B, N = 2, 6
    h = torch.randn(B, N, 24, dtype=torch.float64)
    fr = identity_frames(B, L.H, N, L.a, L.b)
    out, _ = L(h, Frames(fr.dials.double(), fr.gyros.double()))
    q, k, v = L.qkv(h).view(B, N, 3, L.H, L.hd).permute(2, 0, 3, 1, 4)
    ref = L.out(((q @ k.transpose(-1, -2) / math.sqrt(L.hd)).softmax(-1) @ v).transpose(1, 2).reshape(B, N, -1))
    assert torch.allclose(out, ref, atol=1e-12)


def test_rope_only_is_exactly_rope():
    """With no dials or gyros, the layer is ordinary attention with RoPE on the rope channels."""
    L = layer(fingerprint_dim=0, n_dial=0, n_gyro=0, n_free=0, n_rope=4)
    B, N = 2, 9
    h = torch.randn(B, N, 24, dtype=torch.float64)
    angles = torch.arange(N, dtype=torch.float64)[:, None] * 10000.0 ** (-torch.arange(4.0) / 4)
    out, _ = L(h, identity_frames(B, L.H, N, 0, 0), angles[None, None])
    q, k, v = L.qkv(h).view(B, N, 3, L.H, L.hd).permute(2, 0, 3, 1, 4)
    rot = torch.polar(torch.ones_like(angles), angles)
    rope = lambda x: torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (4, 2)).contiguous()) * rot).flatten(-2)
    A = (rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(L.hd)).softmax(-1)
    ref = L.out((A @ v).transpose(1, 2).reshape(B, N, -1))
    assert torch.allclose(out, ref, atol=1e-12)


def test_moving_everything_together_changes_nothing():
    """Turn every token's frame by the same rotation (a push that moves everything together):
    relative frames don't change, so outputs are identical and new frames carry the same turn."""
    L = layer()
    B, N = 2, 6
    h, fp = torch.randn(B, N, 24, dtype=torch.float64), torch.randn(B, N, 5, dtype=torch.float64)
    fr, rope = random_frames(B, L.H, N, L.a, L.b), torch.randn(1, 1, N, L.nr, dtype=torch.float64)
    shift = torch.randn(1, L.H, 1, L.a, dtype=torch.float64)
    P = so3_exp(torch.randn(1, L.H, 1, L.b, 3, dtype=torch.float64))
    out1, new1 = L(h, fr, rope, fp)
    out2, new2 = L(h, Frames(fr.dials + shift, P @ fr.gyros), rope, fp)
    assert torch.allclose(out1, out2, atol=1e-10)
    assert torch.allclose(new1.dials + shift, new2.dials, atol=1e-10)
    assert torch.allclose(P @ new1.gyros, new2.gyros, atol=1e-10)


def test_padding_tokens_do_not_leak():
    L = layer()
    B, N = 1, 8
    h, fp = torch.randn(B, N, 24, dtype=torch.float64), torch.randn(B, N, 5, dtype=torch.float64)
    fr, rope = random_frames(B, L.H, N, L.a, L.b), torch.zeros(1, 1, N, L.nr, dtype=torch.float64)
    keep = torch.ones(B, N, dtype=torch.bool)
    keep[:, 5:] = False
    h2 = h.clone()
    h2[:, 5:] += 3.0
    o1, n1 = L(h, fr, rope, fp, keep[:, None, None, :])
    o2, n2 = L(h2, fr, rope, fp, keep[:, None, None, :])
    assert torch.allclose(o1[:, :5], o2[:, :5], atol=1e-12)
    assert torch.allclose(n1.dials[:, :, :5], n2.dials[:, :, :5], atol=1e-12)


@pytest.mark.parametrize("sync,moves", [(True, False), (False, True)])
def test_switching_forces_off(sync, moves):
    torch.manual_seed(3)
    L = layer(use_sync=sync, use_moves=moves)
    B, N = 2, 5
    h, fp = torch.randn(B, N, 24, dtype=torch.float64), torch.randn(B, N, 5, dtype=torch.float64)
    fr, rope = random_frames(B, L.H, N, L.a, L.b), torch.randn(1, 1, N, L.nr, dtype=torch.float64)
    full = layer()
    full.load_state_dict(L.state_dict(), strict=False)
    with torch.no_grad():  # the switched-off force has rate 0 in the full layer
        if not sync:
            full.sync_rate.fill_(-60.0)
        if not moves:
            full.move_rate.fill_(-60.0)
    out, new = L(h, fr, rope, fp)
    out_f, new_f = full(h, fr, rope, fp)
    assert torch.allclose(out, out_f, atol=1e-12)
    assert torch.allclose(new.dials, new_f.dials, atol=1e-10)
    assert torch.allclose(new.gyros, new_f.gyros, atol=1e-10)


def test_bf16_autocast_keeps_frames_float32_and_rotations_exact():
    L = layer(dtype=torch.float32)
    B, N = 2, 6
    h, fp = torch.randn(B, N, 24), torch.randn(B, N, 5)
    fr = random_frames(B, L.H, N, L.a, L.b, torch.float32)
    rope = torch.randn(1, 1, N, L.nr)
    out32, new32 = L(h, fr, rope, fp)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out16, new16 = L(h, fr, rope, fp)
    assert new16.dials.dtype == torch.float32 and new16.gyros.dtype == torch.float32
    err = (new16.gyros.transpose(-1, -2) @ new16.gyros - torch.eye(3)).abs().max()
    assert err < 1e-5
    assert (out16.float() - out32).abs().max() < 0.05
    assert (new16.dials - new32.dials).abs().max() < 0.05


def kope_setup():
    torch.manual_seed(0)
    B, H, N, hd = 2, 2, 6, 8
    x = torch.randn(B, N, H * hd, dtype=torch.float64)
    phi = torch.randn(B, H, N, hd // 2, dtype=torch.float64) * 3
    return B, H, N, hd, x, phi, Frames(phi, torch.zeros(B, H, N, 0, 3, 3, dtype=torch.float64))


def test_kope_attention_matches_complex_rotation():
    B, H, N, hd, x, phi, frames = kope_setup()
    L = KoPEAttention(H * hd, H).double()
    out, _ = L(x, frames)
    q, k, v = L.qkv(x).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
    z = torch.complex(torch.cos(phi) @ L.mix[None], torch.sin(phi) @ L.mix[None])
    z = z / z.abs()
    rot = lambda t: torch.view_as_complex(t.unflatten(-1, (hd // 2, 2)).contiguous()) * z
    qz, kz, vz = rot(q), rot(k), rot(v)
    A = ((qz.real @ kz.real.transpose(-1, -2) + qz.imag @ kz.imag.transpose(-1, -2)) / math.sqrt(hd)).softmax(-1)
    o = torch.view_as_real((A.to(vz.dtype) @ vz) * z.conj()).flatten(-2)
    assert torch.allclose(out, L.out(o.transpose(1, 2).reshape(B, N, H * hd)), atol=1e-10)


def test_kope_coupling_matches_reference_update():
    """The angle update must equal the reference's project, step and renormalize on (cos, sin)."""
    B, H, N, hd, x, phi, frames = kope_setup()
    C = KoPECoupling(H * hd, H).double()
    new = C(x, frames).dials
    q = F.rms_norm(C.q_proj(x).view(B, N, H, hd).transpose(1, 2), (hd,)) * C.qk_scale
    k = F.rms_norm(C.k_proj(x).view(B, N, H, hd).transpose(1, 2), (hd,))
    J = (q @ k.transpose(-1, -2) / math.sqrt(hd)).softmax(-1)
    delta = (J @ torch.cat((torch.cos(phi), torch.sin(phi)), -1)).reshape(B, H, N, 2, hd // 2)
    pair = torch.stack((torch.cos(phi), torch.sin(phi)), 3)
    delta = delta - (delta * pair).sum(3, keepdim=True) * pair
    updated = F.normalize(pair + C.gamma * delta, dim=3)
    assert torch.allclose(torch.cos(new), updated[..., 0, :], atol=1e-10)
    assert torch.allclose(torch.sin(new), updated[..., 1, :], atol=1e-10)


def test_sync_turns_attention_groups_into_tags():
    """Two groups of tokens that attend within their group, dials starting at random angles.
    Repeated sync steps give each group one tag, with the two tags pushed half a turn apart."""
    torch.manual_seed(7)
    N, dim = 8, 4
    L = FrameAttention(dim, n_heads=1, n_free=2, n_rope=0, n_dial=1, n_gyro=0, sync_init=0.5, use_moves=False)
    with torch.no_grad():
        L.qkv.weight.zero_()
        W = L.qkv.weight.view(3, L.hd, dim)
        W[0, :2, :2] = 3 * torch.eye(2)  # queries <- content
        W[1, :2, :2] = 3 * torch.eye(2)  # keys    <- content
    h = torch.zeros(1, N, dim)
    h[0, :4, 0], h[0, 4:, 1] = 1.0, 1.0  # group A, group B
    fr = Frames(torch.rand(1, 1, N, 1) * 2 * math.pi - math.pi, torch.zeros(1, 1, N, 0, 3, 3))
    for _ in range(60):
        _, fr = L(h, fr)
    z = torch.polar(torch.ones(N), fr.dials.view(N))
    zA, zB = z[:4].mean(), z[4:].mean()
    assert zA.abs() > 0.99 and zB.abs() > 0.99
    assert (zA / zB).angle().abs() > 2.5
