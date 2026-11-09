"""
Measures of whether the frames actually group tokens into objects.

    patch_labels    ground-truth object id of each image patch (from the id mask)
    tag_gap         mean similarity of frames for patch pairs on the same object minus
                    pairs on different objects (dials: cos of the angle difference;
                    gyros: tr(R_i^T R_j) / 3). Accurate tags: same ~ +1, different lower.
    fg_ari          cluster patches (k-means) and score against the true objects with the
                    adjusted Rand index over foreground patches (FG-ARI, 1 = perfect)
    sync            order parameter |mean e^{i phi}| over the object (foreground) patches of an
                    image: near 1 in every image means all objects share one tag (collapse)
"""

import numpy as np
import torch
import torch.nn.functional as F


def patch_labels(masks, patch, min_frac=0.5):
    """masks (B, H, W) object ids -> (B, P) majority id per patch, or -1 if no id covers min_frac."""
    B, H, W = masks.shape
    gh, gw = H // patch, W // patch
    m = masks[:, :gh * patch, :gw * patch].reshape(B, gh, patch, gw, patch).permute(0, 1, 3, 2, 4)
    m = m.reshape(B, gh * gw, patch * patch).long()
    counts = F.one_hot(m, int(masks.max()) + 1).sum(2)
    top, arg = counts.max(-1)
    return torch.where(top >= min_frac * patch * patch, arg, torch.full_like(arg, -1))


def _pair_means(sim, labels):
    fg = labels > 0
    both = fg[:, :, None] & fg[:, None, :]
    eye = torch.eye(labels.shape[1], dtype=torch.bool, device=labels.device)
    same = both & (labels[:, :, None] == labels[:, None, :]) & ~eye
    diff = both & (labels[:, :, None] != labels[:, None, :])
    s = (sim * same).sum() / same.sum().clamp_min(1)
    d = (sim * diff).sum() / diff.sum().clamp_min(1)
    return s.item(), d.item()


def frame_features(dials, gyros):
    """dials (B, H, P, a), gyros (B, H, P, b, 3, 3) -> per-patch features whose dot products are
    the average dial cosine and gyro trace similarity; each part has unit norm per patch."""
    feats = []
    B, H, P = dials.shape[:3]
    if dials.shape[-1]:
        n = H * dials.shape[-1]
        d = dials.permute(0, 2, 1, 3).reshape(B, P, -1)
        feats.append(torch.cat((torch.cos(d), torch.sin(d)), -1) / n ** 0.5)
    if gyros.shape[3]:
        n = H * gyros.shape[3]
        feats.append(gyros.permute(0, 2, 1, 3, 4, 5).reshape(B, P, -1) / (3 * n) ** 0.5)
    return torch.cat(feats, -1) if feats else None


def tag_gaps(dials, gyros, labels):
    out = {}
    if dials.shape[-1]:
        B, H, P, a = dials.shape
        d = dials.permute(0, 2, 1, 3).reshape(B, P, -1)
        sim = (torch.cos(d) @ torch.cos(d).transpose(1, 2) + torch.sin(d) @ torch.sin(d).transpose(1, 2)) / (H * a)
        out["dial_same"], out["dial_diff"] = _pair_means(sim, labels)
        out["tag_gap"] = out["dial_same"] - out["dial_diff"]
    if gyros.shape[3]:
        B, H, P, b = gyros.shape[:4]
        g = gyros.permute(0, 2, 1, 3, 4, 5).reshape(B, P, -1)
        sim = g @ g.transpose(1, 2) / (3 * H * b)
        out["gyro_same"], out["gyro_diff"] = _pair_means(sim, labels)
        out["gyro_gap"] = out["gyro_same"] - out["gyro_diff"]
    return out


def sync_order(dials, gyros, labels):
    """Order parameter over each image's foreground patches, averaged over images, heads, blocks."""
    out = {}
    fg = (labels > 0).float()[:, None, :, None]  # (B, 1, P, 1)
    n = fg.sum(2).clamp_min(1)
    if dials.shape[-1]:
        z = torch.polar(torch.ones_like(dials), dials)
        out["dial_sync"] = ((z * fg).sum(2) / n).abs().mean().item()
    if gyros.shape[3]:
        mean_R = (gyros * fg[..., None, None]).sum(2) / n[..., None, None]
        out["gyro_sync"] = (mean_R.flatten(-2).norm(dim=-1) / 3 ** 0.5).mean().item()
    return out


def kmeans(X, k, iters=30, restarts=3, seed=0):
    """Batched k-means. X (B, P, D) -> labels (B, P), keeping the restart with lowest inertia."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, P, D = X.shape
    best_lab, best_in = None, None
    for _ in range(restarts):
        # k-means++ initialisation, vectorised over the batch
        idx = torch.randint(P, (B,), generator=g).to(X.device)
        C = X[torch.arange(B, device=X.device), idx][:, None]
        for _ in range(1, k):
            d2 = torch.cdist(X, C).min(-1).values ** 2
            p = (d2 + 1e-12) / (d2 + 1e-12).sum(-1, keepdim=True)
            nxt = torch.multinomial(p.cpu(), 1, generator=g).to(X.device)[:, 0]
            C = torch.cat((C, X[torch.arange(B, device=X.device), nxt][:, None]), 1)
        for _ in range(iters):
            lab = torch.cdist(X, C).argmin(-1)
            oh = F.one_hot(lab, k).to(X.dtype)
            cnt = oh.sum(1)
            newC = oh.transpose(1, 2) @ X / cnt.clamp_min(1)[..., None]
            C = torch.where(cnt[..., None] > 0, newC, C)
        lab = torch.cdist(X, C).argmin(-1)
        inertia = ((X - torch.gather(C, 1, lab[..., None].expand(-1, -1, D))) ** 2).sum((1, 2))
        if best_lab is None:
            best_lab, best_in = lab, inertia
        else:
            better = inertia < best_in
            best_lab = torch.where(better[:, None], lab, best_lab)
            best_in = torch.where(better, inertia, best_in)
    return best_lab


def ari(true, pred):
    """Adjusted Rand index of two labelings (1D int arrays), same conventions as scikit-learn."""
    _, t = np.unique(true, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    n = len(t)
    if n <= 1:
        return 1.0
    C = np.zeros((t.max() + 1, p.max() + 1), np.int64)
    np.add.at(C, (t, p), 1)
    sum_sq = (C.astype(np.float64) ** 2).sum()
    n_c, n_k = C.sum(1).astype(np.float64), C.sum(0).astype(np.float64)
    tp = sum_sq - n
    fp = (C @ n_k).sum() - sum_sq
    fn = (C.T @ n_c).sum() - sum_sq
    tn = n * n - fp - fn - sum_sq
    if fn == 0 and fp == 0:
        return 1.0
    return float(2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))


def fg_ari(features, labels, k):
    """Mean FG-ARI over images: cluster all patches into k groups, score foreground patches only."""
    pred = kmeans(features.float(), k).cpu().numpy()
    lab = labels.cpu().numpy()
    scores = []
    for i in range(len(lab)):
        fg = lab[i] > 0
        if fg.sum() >= 2:
            scores.append(ari(lab[i][fg], pred[i][fg]))
    return float(np.mean(scores)) if scores else float("nan")
