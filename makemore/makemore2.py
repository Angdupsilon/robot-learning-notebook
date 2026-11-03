# Nov 3, 2026. Karpathy's makemore part 2: MLP

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# import names
words = open("names.txt", "r").read().splitlines()

# index characters + look-up table
chars = sorted(list(set("".join(words)))) # a - z, 0 -25
stoi = {s:i+1 for i, s in enumerate(chars)} # create dict for letter and value (index)
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}

# build the dataset
block_size = 3

def build_dataset(words):  #context length: how many characters we take to predict the next one
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size # pad "." before word
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # rolling context

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  # print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
k1 = int(0.8*len(words))
k2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:k1])
Xeval, Yeval = build_dataset(words[k1:k2])
Xte, Yte = build_dataset(words[k2:])

# initialize configs
g = torch.Generator().manual_seed(21391991)
C = torch.randn((27, 5), requires_grad=True, generator=g) # two-dimensional gradient for each of the letters, NxD

W1 = torch.randn((15, 200), requires_grad=True, generator=g) # input as 6 values (3 letters * 2 dimensions)
b1 = torch.randn((200), requires_grad=True, generator=g)

W2 = torch.randn((200, 27), requires_grad=True, generator=g)
b2 = torch.randn((27), requires_grad=True, generator=g)

# bundle together for easier backward pass
parameters = [C, W1, b1, W2, b2]

# testing learning rates
runs = 30000
lre = torch.linspace(-3, 0, runs)
lrs = 10**lre

# track to find best learning rate
lri = []
lossi = []

# training!
for i in range(runs):

    # minibatches!!!
    ix = torch.randint(0, Xtr.shape[0], (32,)) #arguments: lowest, highest, output size

    #forward pass
    emb = C[Xtr[ix]]  # extracting gradients for the inputs
    h1 = torch.reshape(emb, (-1,15)) @ W1 + b1 # flatten and calculate layer
    n1 = torch.tanh(h1) # normalize

    h2 = n1 @ W2 + b2
    n2 = torch.softmax(h2, dim=1) # return possibilities of outputs
    loss = F.cross_entropy(h2, Ytr[ix]) # calculates loss

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update weights
    for p in parameters:
        # lr = lrs[i]
        lr = 0.1
        p.data += -lr * p.grad

    # track lr and loss
    lri.append(lrs[i])
    lossi.append(loss.item())

    #
    # print(loss.item())

# find best lr and display
'''plt.plot(lri, lossi)
plt.show()'''

# Test loss
emb = C[Xtr]  # extracting gradients for the inputs
h1 = torch.reshape(emb, (-1,15)) @ W1 + b1 # flatten and calculate layer
n1 = torch.tanh(h1) # normalize

h2 = n1 @ W2 + b2
n2 = torch.softmax(h2, dim=1) # return possibilities of outputs
loss = F.cross_entropy(h2, Ytr) # calculates loss

print(loss.item())

# Eval loss
emb = C[Xeval]  # extracting gradients for the inputs
h1 = torch.reshape(emb, (-1,15)) @ W1 + b1 # flatten and calculate layer
n1 = torch.tanh(h1) # normalize

h2 = n1 @ W2 + b2
n2 = torch.softmax(h2, dim=1) # return possibilities of outputs
loss = F.cross_entropy(h2, Yeval) # calculates loss

print(loss.item())


for i in range(20):

    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])] # (1, block_size, d)
        h1 = torch.reshape(emb, (-1, 15)) @ W1 + b1  # flatten and calculate layer
        n1 = torch.tanh(h1)  # normalize

        h2 = n1 @ W2 + b2
        n2 = torch.softmax(h2, dim=1)  # return possibilities of outputs

        ix = torch.multinomial(n2, num_samples=1, replacement=True, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))