"""
Checks that the fast Frame Attention code does what the math says.

The main test rebuilds every pairwise relative frame T_ij = g_i^T g_j explicitly
(the slow N^2 way) and confirms the fast "rotate once, one attention call" version
gives identical outputs and identical frame updates.

Run:  python test_frame_attention.py      (or: pytest test_frame_attention.py)
"""

import math

import torch
import torch.nn.functional as F

from frame_attention import (
    FrameAttention, FrameBlock, FrameInit, Frames, hat, reorthonormalize, so3_exp, vee,
)

torch.set_default_dtype(torch.float64)


def random_frames(B, H, N, a, b, scale=2.0):
    return Frames(torch.randn(B, H, N, a) * scale, so3_exp(torch.randn(B, H, N, b, 3) * scale))


def make_layer(**kw):
    torch.manual_seed(0)
    cfg = dict(dim=24, n_heads=3, n_free=2, n_dials=2, n_gyros=2, fingerprint_dim=5, fp_head_dim=4)
    cfg.update(kw)
    layer = FrameAttention(**cfg)
    with torch.no_grad():  # make every mechanism matter: real spins, real frustration
        layer.spin.weight.normal_(0, 0.3)
        layer.alpha.normal_(0, 0.5)
        layer.f_gyro.normal_(0, 0.5)
        layer.eta_sync.normal_(0, 0.5)
        layer.eta_comp.normal_(0, 0.5)
    return layer


# --------------------------------------------------------------------------- #
# Slow reference: explicit T_ij for every pair
# --------------------------------------------------------------------------- #
def naive_forward(L, h, frames, fp=None, causal=False):
    B, N, _ = h.shape
    H, nf, a, b, hd = L.H, L.nf, L.a, L.b, L.hd
    q, k, v = L.qkv(h).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
    spin = L.spin(h).view(B, N, H, a + 3 * b).transpose(1, 2)
    theta, w = spin[..., :a], spin[..., a:].unflatten(-1, (b, 3))
    phi, R = frames.dials, frames.gyros

    # Full block-diagonal frame matrix per token: G_i = diag(I_free, R(phi_1..a), R_1..b)
    G = torch.zeros(B, H, N, hd, hd)
    G[..., :nf, :nf] = torch.eye(nf)
    for d in range(a):
        c, s = torch.cos(phi[..., d]), torch.sin(phi[..., d])
        i = nf + 2 * d
        G[..., i, i], G[..., i, i + 1], G[..., i + 1, i], G[..., i + 1, i + 1] = c, -s, s, c
    for g in range(b):
        i = nf + 2 * a + 3 * g
        G[..., i:i + 3, i:i + 3] = R[..., g, :, :]

    T = torch.einsum("bhiyx,bhjyz->bhijxz", G, G)  # T_ij = G_i^T G_j   (B,H,N,N,hd,hd)
    s = torch.einsum("bhix,bhijxz,bhjz->bhij", q, T, k) / math.sqrt(hd)
    if fp is not None:
        f = L.fp(fp).view(B, N, H, L.fph).transpose(1, 2)
        s = s + F.softplus(L.lam).view(1, H, 1, 1) * torch.einsum("bhic,bhjc->bhij", f, f)
    allowed = torch.ones(N, N, dtype=torch.bool).tril() if causal else torch.ones(N, N, dtype=torch.bool)
    A = s.masked_fill(~allowed, float("-inf")).softmax(-1)
    u = allowed.double() / allowed.double().sum(-1, keepdim=True)  # uniform baseline u_ij
    Wr = A - L.repel * u  # (A_ij - kappa u_ij)

    o = torch.einsum("bhij,bhijxz,bhjz->bhix", A, T, v)  # sum_j A_ij T_ij v_j
    out = L.out(o.transpose(1, 2).reshape(B, N, H * hd))

    eta_s, eta_c = F.softplus(L.eta_sync), F.softplus(L.eta_comp)
    # dials: sum_j (A_ij - k u_ij) sin(phi_j - phi_i - alpha)  +  sum_j A_ij theta_j
    dphi = phi.unsqueeze(-3) - phi.unsqueeze(-2) - L.alpha.view(1, H, 1, 1, a)  # [i, j]
    sync_d = torch.einsum("bhij,bhija->bhia", Wr, torch.sin(dphi))
    comp_d = torch.einsum("bhij,bhja->bhia", A, theta)
    phi_new = phi + eta_s[:, :a].view(1, H, 1, a) * sync_d + eta_c[:, :a].view(1, H, 1, a) * comp_d
    # gyros: vee( sum_j (A - k u) T_ij F^T )  +  vee( sum_j A_ij T_ij [w_j]_x T_ij^T )
    Tg = torch.einsum("bhigyx,bhjgyz->bhijgxz", R, R)
    Fg = so3_exp(L.f_gyro)
    sync_g = vee(torch.einsum("bhij,bhijgxy->bhigxy", Wr, Tg) @ Fg.transpose(-1, -2).unsqueeze(1))
    conj = Tg @ hat(w).unsqueeze(2) @ Tg.transpose(-1, -2)
    comp_g = vee(torch.einsum("bhij,bhijgxy->bhigxy", A, conj))
    xi = eta_s[:, a:].view(1, H, 1, b, 1) * sync_g + eta_c[:, a:].view(1, H, 1, b, 1) * comp_g
    R_new = reorthonormalize(R @ so3_exp(xi))
    return out, Frames(phi_new, R_new)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_fast_equals_naive_pairwise():
    L = make_layer()
    torch.manual_seed(1)
    B, N = 2, 7
    h, fp = torch.randn(B, N, 24), torch.randn(B, N, 5)
    fr = random_frames(B, L.H, N, L.a, L.b)
    for causal in (False, True):
        out, new, _ = L(h, fr, fp, is_causal=causal)
        out_n, new_n = naive_forward(L, h, fr, fp, causal)
        assert torch.allclose(out, out_n, atol=1e-10), (out - out_n).abs().max()
        assert torch.allclose(new.dials, new_n.dials, atol=1e-10), (new.dials - new_n.dials).abs().max()
        assert torch.allclose(new.gyros, new_n.gyros, atol=1e-10), (new.gyros - new_n.gyros).abs().max()


def test_identity_frames_is_standard_attention():
    L = make_layer(fingerprint_dim=0)
    torch.manual_seed(2)
    B, N = 2, 6
    h = torch.randn(B, N, 24)
    ident = Frames(torch.zeros(B, L.H, N, L.a), torch.eye(3).expand(B, L.H, N, L.b, 3, 3).clone())
    out, _, _ = L(h, ident)
    q, k, v = L.qkv(h).view(B, N, 3, L.H, L.hd).permute(2, 0, 3, 1, 4)
    A = (q @ k.transpose(-1, -2) / math.sqrt(L.hd)).softmax(-1)
    ref = L.out((A @ v).transpose(1, 2).reshape(B, N, -1))
    assert torch.allclose(out, ref, atol=1e-12)


def test_rope_is_a_special_case():
    """Dials set from positions, values not transported -> exactly RoPE (interleaved pairs)."""
    torch.manual_seed(3)
    dim, H, a, N, B = 16, 2, 4, 9, 2
    L = FrameAttention(dim, n_heads=H, n_free=0, n_dials=a, n_gyros=0, transport_values=False)
    h = torch.randn(B, N, dim)
    frames = FrameInit(dim, H, a, 0)(h, positions=torch.arange(N))
    out, _, _ = L(h, frames)

    q, k, v = L.qkv(h).view(B, N, 3, H, 2 * a).permute(2, 0, 3, 1, 4)
    freq = 10000.0 ** (-torch.arange(a) / a)
    rot = torch.polar(torch.ones(N, a), torch.arange(N).view(N, 1) * freq)  # e^{i pos freq}
    rope = lambda x: torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (a, 2)).contiguous()) * rot).flatten(-2)
    A = (rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(2 * a)).softmax(-1)
    ref = L.out((A @ v).transpose(1, 2).reshape(B, N, -1))
    assert torch.allclose(out, ref, atol=1e-12)


def test_world_rotation_equivariance():
    """Rotate every token's frame by the same P (a 'poke' that moves everything together):
    relative frames T_ij don't change, so outputs are identical and new frames just carry P."""
    L = make_layer()
    torch.manual_seed(4)
    B, N = 2, 6
    h, fp = torch.randn(B, N, 24), torch.randn(B, N, 5)
    fr = random_frames(B, L.H, N, L.a, L.b)
    shift = torch.randn(1, L.H, 1, L.a)
    P = so3_exp(torch.randn(1, L.H, 1, L.b, 3))
    moved = Frames(fr.dials + shift, P @ fr.gyros)
    out1, new1, _ = L(h, fr, fp)
    out2, new2, _ = L(h, moved, fp)
    assert torch.allclose(out1, out2, atol=1e-10)
    assert torch.allclose(new1.dials + shift, new2.dials, atol=1e-10)
    assert torch.allclose(P @ new1.gyros, new2.gyros, atol=1e-10)


def test_causal_has_no_future_leak():
    L = make_layer()
    torch.manual_seed(5)
    B, N, cut = 1, 8, 5
    h, fp = torch.randn(B, N, 24), torch.randn(B, N, 5)
    fr = random_frames(B, L.H, N, L.a, L.b)
    h2, fp2 = h.clone(), fp.clone()
    h2[:, cut:] += torch.randn(B, N - cut, 24)
    fp2[:, cut:] += 1.0
    fr2 = Frames(fr.dials.clone(), fr.gyros.clone())
    fr2.dials[:, :, cut:] += 1.0
    o1, n1, _ = L(h, fr, fp, is_causal=True)
    o2, n2, _ = L(h2, fr2, fp2, is_causal=True)
    assert torch.allclose(o1[:, :cut], o2[:, :cut], atol=1e-12)
    assert torch.allclose(n1.dials[:, :, :cut], n2.dials[:, :, :cut], atol=1e-12)
    assert torch.allclose(n1.gyros[:, :, :cut], n2.gyros[:, :, :cut], atol=1e-12)


def test_explicit_masks_match_builtin_and_empty_blocks_work():
    for kw in (dict(), dict(n_dials=0), dict(n_gyros=0), dict(n_free=0)):
        L = make_layer(**kw)
        torch.manual_seed(8)
        B, N = 2, 6
        h, fp = torch.randn(B, N, 24), torch.randn(B, N, 5)
        fr = random_frames(B, L.H, N, L.a, L.b)
        full = torch.ones(N, N, dtype=torch.bool)
        for mask, causal in ((full, False), (full.tril(), False), (full, True)):
            o1, n1, _ = L(h, fr, fp, attn_mask=mask, is_causal=causal)
            o2, n2, _ = L(h, fr, fp, is_causal=causal or not mask.all())
            assert torch.allclose(o1, o2, atol=1e-12) and torch.allclose(n1.dials, n2.dials, atol=1e-12)
            assert torch.allclose(n1.gyros, n2.gyros, atol=1e-12)


def test_deep_stack_stays_on_rotations_and_trains():
    torch.manual_seed(6)
    dim, H, a, b = 32, 4, 3, 2
    init = FrameInit(dim, H, a, b)
    with torch.no_grad():
        init.proj.weight.normal_(0, 0.5)
    blocks = [FrameBlock(dim, n_heads=H, n_free=2, n_dials=a, n_gyros=b, steps=2) for _ in range(12)]
    x = torch.randn(2, 10, dim)
    frames = init(x)
    for blk in blocks:
        x, frames, _ = blk(x, frames)
    err = (frames.gyros.transpose(-1, -2) @ frames.gyros - torch.eye(3)).abs().max()
    assert err < 1e-8, err
    x.square().mean().backward()
    for blk in blocks:
        for name, p in blk.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
    assert blocks[0].attn.eta_sync.grad.abs().sum() > 0
    assert blocks[0].attn.spin.weight.grad.abs().sum() > 0


def test_gradients_finite_at_identity_frames():
    L = make_layer()
    B, N = 1, 5
    h = torch.randn(B, N, 24, requires_grad=True)
    ident = Frames(torch.zeros(B, L.H, N, L.a), torch.eye(3).expand(B, L.H, N, L.b, 3, 3).clone())
    out, new, _ = L(h, ident, torch.randn(B, N, 5))
    (out.sum() + new.dials.sum() + new.gyros.sum()).backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in L.parameters() if p.grad is not None)


def test_sync_turns_attention_groups_into_tags():
    """Two groups of tokens that attend within their group. Starting from random dial
    angles, the sync rule should give each group one shared tag, with the two tags
    pushed far apart by the repulsion term."""
    torch.manual_seed(7)
    N, dim = 8, 4
    L = FrameAttention(dim, n_heads=1, n_free=2, n_dials=1, n_gyros=0, eta_sync=0.5,
                       frustration=False, steps=60)
    with torch.no_grad():
        L.qkv.weight.zero_()
        Wq = L.qkv.weight.view(3, L.hd, dim)
        Wq[0, :2, :2] = 3 * torch.eye(2)  # query free channels <- content
        Wq[1, :2, :2] = 3 * torch.eye(2)  # key free channels   <- content
        L.spin.weight.zero_()
    h = torch.zeros(1, N, dim)
    h[0, :4, 0], h[0, 4:, 1] = 1.0, 1.0  # group A, group B
    start = Frames(torch.rand(1, 1, N, 1) * 2 * math.pi - math.pi, torch.zeros(1, 1, N, 0, 3, 3))
    _, end, _ = L(h, start)
    z = torch.polar(torch.ones(N), end.dials.view(N))
    zA, zB = z[:4].mean(), z[4:].mean()
    gap = (zA / zB).angle().abs()
    print(f"\n  group coherence A={zA.abs():.4f} B={zB.abs():.4f}, tag gap = {gap:.3f} rad (pi = 3.142)")
    assert zA.abs() > 0.99 and zB.abs() > 0.99
    assert gap > 2.5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
