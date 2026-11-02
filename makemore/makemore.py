'''
We are going to build a name predictor, letter by letter
'''
from torch._C import dtype

# import words from words list

words = open("names.txt", "r").read().splitlines() # each name as its own line

'''
# because we predict output letter by letter, we always predict by the sequence of two characters: one before and another after
# initialize statistic counting
b = {} # b is dictionary, key as two letter sequences, value as statistical count

for w in words[:]:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        # learn statistics on which one follows the other, just count which letter follows which
        b[bigram] = b.get(bigram, 0) +1
        # print(ch1, ch2)

print(sorted(b.items(), key = lambda kv:-kv[1])) # we first make dictionary into N x 2 list with characters and values. Then we sort them according to the count
'''



# Now organize data with torch

import torch
N = torch.zeros((27, 27), dtype=torch.int32) # initialize tensor to organize

# index characters + look-up table
chars = sorted(list(set("".join(words)))) # a - z, 0 -25
stoi = {s:i+1 for i, s in enumerate(chars)} # create dict for letter and value (index)
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}

# map every word
for w in words[:]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Set up probability by normalize the entire matrix
P = N.float()
P /= P.sum(1, keepdim=True)

# sampling using pytorch multinomial
g = torch.Generator().manual_seed(21301181) # not truly random but random seed setup

ix = 0 # initialize ix at row one ("." or start)
out = []
while True:
    # p = N[ix].float()
    # p /= p.sum()
    p = P[ix]

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        break

print("".join(out))


# Evaluate model, model == grid/matrix of inputs (probabilities of two pairings)

log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood # make value positive to find lowest loss
print(f'{nll=}')
print(f'{nll/n}')

# training dataset
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs) # input, AKA initial letter code
ys = torch.tensor(ys) # output, AKA the following letter code

# one-hot encoding
import torch.nn.functional as F

x_enc = F.one_hot(xs, num_classes=27).float()

# initialize Weights
W = torch.randn((27, 27), generator=g, requires_grad=True) # input 27 inputs, and give 27 output options (AKA finding the next value). G is manual seed defined above

for k in range(100):
    # Because output is between - and +, we softmax them to be a probability (between 0 and 1)
    logits = x_enc @ W # matrix multiply, as in neural net
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # negative log likelihood as featured above, compared at each index with the correct output
    loss = -probs[torch.arange(5), ys].log().mean() + 0.01*(W**2).mean() # regularization: loss gets larger when weights are none-zero, so the weights are pushed towards zero to achieve a more uniform gradient compared to without
    print(loss)


    W.grad = None
    loss.backward()
    W.data += -10 * W.grad

# finally, sample from the 'neural net' model
for i in range(5):

    out = []
    ix = 0
    while True:
        # ----------
        # BEFORE:
        # p = P[ix]
        # ----------
        # NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
        # ----------

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

'''
Notes:
Two approaches align. We started with statistical and counting approach, then we coded a neural network approach.
However, neural networks allows for better scaling. 
'''