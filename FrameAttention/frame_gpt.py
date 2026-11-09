"""
Frame GPT: the gpt.py from Karpathy's "Let's build GPT" lecture, with Frame Attention.
This is the short version for reading; experiments use frame_attention.py + train.py.

Next to its embedding, every token now carries a compass ("frame") in each head:
  dials - 2D rotations (angles)   -> tags: which thing do I belong to?
  gyros - 3D rotations (matrices) -> structure: order and direction matter
Attention compares tokens after translating them into each other's frame.
After each layer, tokens that attend to each other align their frames (sync),
and tokens pass each other small turns (moves).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# frame attention: each head's 64 channels = 16 plain + 12 dials x 2 + 8 gyros x 3
n_free = 16 # channels that never rotate, so separate things can still talk
n_dial = 12 # 2D rotations
n_gyro = 8 # 3D rotations
repel = 1.0 # how hard tokens that ignore each other push their frames apart
# ------------

torch.manual_seed(1337)
assert n_free + 2 * n_dial + 3 * n_gyro == n_embd // n_head

# python download_data.py --only tinyshakespeare   (reuses ../nanoGPT/ShakespeareText.txt)
with open('data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters in the text, and a mapping to and from integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # list of integers -> string

# train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # a small batch of inputs x and targets y (y is x shifted by one)
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- rotations ----------------

def rot2(x, a):
    # turn each pair of numbers in x (..., n_dial, 2) by its angle in a (..., n_dial)
    c, s = a.cos(), a.sin()
    return torch.stack((c*x[..., 0] - s*x[..., 1], s*x[..., 0] + c*x[..., 1]), dim=-1)

def rot3(x, R):
    # turn each triple in x (..., n_gyro, 3) by its matrix in R (..., n_gyro, 3, 3)
    return (R @ x.unsqueeze(-1)).squeeze(-1)

def hat(w):
    # a spin (axis * speed) as a 3x3 matrix
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), dim=-1).view(*w.shape, 3)

def vee(M):
    # the opposite of hat: read the spin back out of a 3x3 matrix
    return 0.5 * torch.stack((M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]), dim=-1)

def spin_to_rotation(w):
    # "spin at speed w for one second" -> rotation matrix (Rodrigues' formula)
    th2 = (w*w).sum(-1)[..., None, None]
    th = th2.clamp(min=1e-12).sqrt()
    small = th2 < 1e-6 # use the Taylor series near zero so nothing divides by 0
    a = torch.where(small, 1 - th2/6, th.sin()/th)
    b = torch.where(small, 0.5 - th2/24, (1 - th.cos())/th2.clamp(min=1e-12))
    K = hat(w)
    return torch.eye(3, device=w.device) + a*K + b*(K @ K)

# ---------------- model ----------------

class Head(nn.Module):
    """ one head of frame attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.move = nn.Linear(n_embd, n_dial + 3*n_gyro, bias=False) # the turn each token hands out
        self.sync_rate = nn.Parameter(torch.tensor(0.3)) # how fast frames align
        self.move_rate = nn.Parameter(torch.tensor(0.1)) # how strongly turns get applied
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def turn(self, x, dial, gyro, back=False):
        # rotate the dial and gyro channels of x (B,T,hs) by each token's frame (or undo it)
        B, T, _ = x.shape
        free, d, g = x.split([n_free, 2*n_dial, 3*n_gyro], dim=-1)
        d = rot2(d.view(B, T, n_dial, 2), -dial if back else dial)
        g = rot3(g.view(B, T, n_gyro, 3), gyro.transpose(-2, -1) if back else gyro)
        return torch.cat([free, d.view(B, T, -1), g.view(B, T, -1)], dim=-1)

    def forward(self, x, dial, gyro):
        # x: (B,T,C)   dial: (B,T,n_dial) angles   gyro: (B,T,n_gyro,3,3) rotations
        B,T,C = x.shape
        # translate keys, queries and values out of each token's own frame into a shared one
        k = self.turn(self.key(x), dial, gyro)   # (B,T,hs)
        q = self.turn(self.query(x), dial, gyro) # (B,T,hs)
        v = self.turn(self.value(x), dial, gyro) # (B,T,hs)
        # attention scores, exactly as in plain attention
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # weighted sum of values, then translate the result back into my own frame
        out = self.turn(self.dropout(wei) @ v, dial, gyro, back=True) # (B, T, hs)

        # --- update the frames ---
        # sync: lean toward the frames of tokens I attend to, away from the average token I can see
        avg = self.tril[:T, :T] / self.tril[:T, :T].sum(-1, keepdim=True) # (T, T)
        pull = wei - repel * avg # (B, T, T)
        dial_sync = (pull @ dial.sin()) * dial.cos() - (pull @ dial.cos()) * dial.sin() # sum_j pull_ij sin(dial_j - dial_i)
        gyro_sync = vee(gyro.transpose(-2, -1) @ (pull @ gyro.reshape(B, T, -1)).view(B, T, n_gyro, 3, 3))
        # moves: collect the turns of tokens I attend to, translated into my frame
        m = self.move(x)
        dial_move = wei @ m[..., :n_dial] # (B, T, n_dial)
        gyro_move = rot3(m[..., n_dial:].view(B, T, n_gyro, 3), gyro) # each move in the shared frame
        gyro_move = rot3((wei @ gyro_move.view(B, T, -1)).view(B, T, n_gyro, 3), gyro.transpose(-2, -1)) # back into mine
        # apply: dial angles just add, gyros get multiplied by a small rotation
        dial = dial + self.sync_rate * dial_sync + self.move_rate * dial_move
        gyro = gyro @ spin_to_rotation(self.sync_rate * gyro_sync + self.move_rate * gyro_move)
        gyro = 1.5*gyro - 0.5 * gyro @ gyro.transpose(-2, -1) @ gyro # snap back to an exact rotation
        return out, dial, gyro

class MultiHeadAttention(nn.Module):
    """ multiple heads of frame attention in parallel, each with its own frames """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dials, gyros):
        # dials: (B, n_head, T, n_dial)   gyros: (B, n_head, T, n_gyro, 3, 3)
        outs, dials, gyros = zip(*[h(x, dials[:, i], gyros[:, i]) for i, h in enumerate(self.heads)])
        out = self.dropout(self.proj(torch.cat(outs, dim=-1)))
        return out, torch.stack(dials, dim=1), torch.stack(gyros, dim=1)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, dials, gyros):
        a, dials, gyros = self.sa(self.ln1(x), dials, gyros) # frames ride along next to x
        x = x + a
        x = x + self.ffwd(self.ln2(x))
        return x, dials, gyros

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.frame_init = nn.Linear(n_embd, n_head * (n_dial + 3*n_gyro)) # each token's starting compass
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        # every frame starts at "no rotation", so the model starts out as a plain GPT
        torch.nn.init.zeros_(self.frame_init.weight)
        torch.nn.init.zeros_(self.frame_init.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        f = self.frame_init(x).view(B, T, n_head, -1).transpose(1, 2) # (B, n_head, T, n_dial + 3*n_gyro)
        dials = f[..., :n_dial] # (B, n_head, T, n_dial)
        gyros = spin_to_rotation(f[..., n_dial:].reshape(B, n_head, T, n_gyro, 3)) # (B, n_head, T, n_gyro, 3, 3)
        for block in self.blocks:
            x, dials, gyros = block(x, dials, gyros)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # last time step only -> (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == '__main__':
    model = GPTLanguageModel()
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
