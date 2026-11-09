"""
The model used in every experiment: a small transformer whose attention layers are FrameAttention.

Inputs are a sequence of tokens:
    image tasks: [CLS] + 4 question tokens + image patches
    text tasks:  [CLS] + words
The answer is read from the CLS token.

Variants (every head has 64 channels; 8 rope pairs = 16 channels in all variants):
    A        rope only. The control: an ordinary transformer with RoPE.
    A_prime  A with extra layers, so its compute matches D. Rules out "it's just more compute".
    B        + dials with sync and moves. Does the update rule help?
    C        + gyros with sync and moves. Do gyros alone help?
    D        + dials and gyros. The full Frame Attention.
    E        D + motion fingerprints (MOVi only). Does common fate help?
"""

import math

import torch
import torch.nn as nn

from frame_attention import FrameAttention, Frames, so3_exp

HEAD_DIM = 64
N_ROPE = 8  # rope pairs per head (16 channels)

#  channels per head:  free  +  rope (16)  +  2 x dials  +  3 x gyros  =  64
VARIANTS = {
    "A":       dict(n_free=48, n_dial=0,  n_gyro=0),
    "A_prime": dict(n_free=48, n_dial=0,  n_gyro=0),
    "B":       dict(n_free=26, n_dial=11, n_gyro=0),
    "C":       dict(n_free=30, n_dial=0,  n_gyro=6),
    "D":       dict(n_free=8,  n_dial=11, n_gyro=6),
    "E":       dict(n_free=8,  n_dial=11, n_gyro=6),
}
for _name, _v in VARIANTS.items():
    assert _v["n_free"] + 2 * N_ROPE + 2 * _v["n_dial"] + 3 * _v["n_gyro"] == HEAD_DIM, _name


def rope_angles(coords, n_rope, base):
    """coords: (N,) for 1D positions or (N, 2) for 2D patch grids -> angles (N, n_rope).
    For 2D, the first half of the pairs encode x and the second half encode y."""
    if coords.dim() == 1:
        freq = base ** (-torch.arange(n_rope, dtype=torch.float32) / n_rope)
        return coords.float()[:, None] * freq
    half = n_rope // 2
    freq = base ** (-torch.arange(half, dtype=torch.float32) / half)
    return torch.cat((coords[:, :1].float() * freq, coords[:, 1:2].float() * freq), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, n_heads, layout, update_frames, fingerprint_dim, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FrameAttention(dim, n_heads, n_rope=N_ROPE, fingerprint_dim=fingerprint_dim,
                                   update_frames=update_frames, **layout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x, frames, rope, fingerprints, attn_mask):
        a, frames = self.attn(self.ln1(x), frames, rope, fingerprints, attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, frames


class FrameTransformer(nn.Module):
    def __init__(self, variant, n_answers, n_heads=4, n_layers=6, *, kind="image", img_size=64, patch=8,
                 n_question=4, vocab_size=32, max_len=128, fingerprint_dim=0, frame_init_std=1.0):
        super().__init__()
        assert variant in VARIANTS, variant
        if variant == "E":
            assert fingerprint_dim > 0, "variant E needs motion fingerprints (task movi_a)"
        else:
            fingerprint_dim = 0  # only E sees motion
        layout = {k: v for k, v in VARIANTS[variant].items()}
        dim = HEAD_DIM * n_heads
        self.variant, self.kind, self.dim, self.H = variant, kind, dim, n_heads
        self.n_dial, self.n_gyro = layout["n_dial"], layout["n_gyro"]
        self.has_frames = self.n_dial + self.n_gyro > 0
        self.fingerprint_dim = fingerprint_dim

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.tok_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.type_embed = nn.Embedding(3, dim)  # 0 = CLS, 1 = question/word, 2 = patch
        if kind == "image":
            g = img_size // patch
            self.patch, self.grid, self.n_prefix = patch, g, 1 + n_question
            self.patch_embed = nn.Conv2d(3, dim, patch, patch)
            self.q_pos = nn.Parameter(torch.zeros(1, n_question, dim))  # which question slot
            ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
            patch_angles = rope_angles(torch.stack((xs.flatten(), ys.flatten()), -1), N_ROPE, base=100.0)
            angles = torch.cat((torch.zeros(self.n_prefix, N_ROPE), patch_angles), 0)  # CLS/question: no rotation
        else:
            self.n_prefix = 1
            angles = rope_angles(torch.arange(1 + max_len), N_ROPE, base=10000.0)
        self.register_buffer("rope", angles[None, None], persistent=False)  # (1, 1, N, n_rope)

        if self.has_frames:
            self.frame_norm = nn.LayerNorm(dim)
            self.frame_init = nn.Linear(dim, n_heads * (self.n_dial + 3 * self.n_gyro))
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, layout, update_frames=(i < n_layers - 1), fingerprint_dim=fingerprint_dim)
            for i in range(n_layers))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_answers)
        self.apply(self._init_weights)
        if self.has_frames:  # starting angles spread ~frame_init_std radians, set by content
            nn.init.normal_(self.frame_init.weight, std=frame_init_std / math.sqrt(dim))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def init_frames(self, x):
        """Starting frames from content: tokens with the same content start with the same tag,
        different content starts spread apart, so the sync rule is active from the first step."""
        B, N, _ = x.shape
        if not self.has_frames:
            return Frames(x.new_zeros(B, self.H, N, 0, dtype=torch.float32),
                          x.new_zeros(B, self.H, N, 0, 3, 3, dtype=torch.float32))
        z = self.frame_init(self.frame_norm(x)).view(B, N, self.H, -1).transpose(1, 2).float()
        dials = z[..., :self.n_dial]
        gyros = so3_exp(z[..., self.n_dial:].unflatten(-1, (self.n_gyro, 3)))
        return Frames(dials, gyros)

    def embed(self, images=None, q_tokens=None, text=None):
        B = (images if images is not None else text).shape[0]
        cls = (self.cls + self.type_embed.weight[0]).expand(B, -1, -1)
        if self.kind == "image":
            p = self.patch_embed(images).flatten(2).transpose(1, 2) + self.type_embed.weight[2]
            q = self.tok_embed(q_tokens) + self.q_pos + self.type_embed.weight[1]
            return torch.cat((cls, q, p), 1), None
        words = self.tok_embed(text) + self.type_embed.weight[1]
        keep = torch.cat((torch.ones_like(text[:, :1], dtype=torch.bool), text != 0), 1)  # CLS + non-pad
        return torch.cat((cls, words), 1), keep[:, None, None, :]

    def forward(self, images=None, q_tokens=None, text=None, fingerprints=None, return_internals=False):
        x, attn_mask = self.embed(images, q_tokens, text)
        N = x.shape[1]
        frames = self.init_frames(x)
        rope = self.rope[:, :, :N]
        fp = None
        if self.fingerprint_dim and fingerprints is not None:  # zeros for CLS / question tokens
            fp = torch.cat((fingerprints.new_zeros(x.shape[0], self.n_prefix, fingerprints.shape[-1]),
                            fingerprints), 1).to(x.dtype)
        frames_per_layer = []
        for blk in self.blocks:
            if return_internals:
                frames_per_layer.append(frames)
            x, frames = blk(x, frames, rope, fp, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x[:, 0])
        if not return_internals:
            return logits
        # frames_per_layer[-1] are the frames the last layer used (it does not update them)
        return logits, dict(frames=frames_per_layer, hidden=x[:, self.n_prefix:])


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        rate = name.endswith(("sync_rate", "move_rate", "lam"))
        (decay if p.dim() >= 2 and not rate else no_decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def forward_flops(model, example):
    """Matmul FLOPs of one forward pass on `example` (a dict of model inputs), per sample."""
    from torch.utils.flop_counter import FlopCounterMode
    was_training = model.training
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(**example)
    model.train(was_training)
    batch = next(v for v in example.values() if torch.is_tensor(v)).shape[0]
    return fc.get_total_flops() / batch


def matched_layers(make_model, n_layers, example):
    """Number of layers that gives variant A at least the forward FLOPs of variant D."""
    target = forward_flops(make_model("D", n_layers), example)
    base = forward_flops(make_model("A", n_layers), example)
    return max(n_layers, math.ceil(n_layers * target / base)), target
