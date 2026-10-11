import torch

# Set Up
torch.manual_seed(0)

x=torch.randn(3, dtype=torch.float64)
y=torch.randn(2, dtype=torch.float64)

print(x, y)
print(x.shape, y.shape)

# Model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 4, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(4, output_dim, dtype=torch.float64)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model=Model(3, 2)

# Training
n_iters = 3

loss = torch.nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(), lr=0.1)

# Run
for iteration in range(n_iters):
    # z1 = fc1(x)
    # h=torch.relu(z1)
    # y_pred=fc2(h)

    y_pred=model(x)

    l = loss(y_pred, y)

    l.backward()

    # # AI added check each parameters
    # for name, p in model.named_parameters():
    #     print(f"{name}.grad =\n{p.grad}")
    # print("loss:", l.item())

    optimizer.step()

    optimizer.zero_grad()

    # See weight changes:
    [w1, b1, w2, b2] = model.parameters()

    print(f"In iteration{iteration+1}, w1={w1[0,0].item():.6f}, b1={b1[0].item():.6f}, w2={w2[0,0].item():.6f}, and b2={b2[0].item():.6f}")
    print(f"Loss={l:.6f}")
    # Question: I just don't understand what w1[0,0] do? Why does it access the value for this?
    # Answer: [0, 0] only a cell in a grid that connects the first input to the first hidden neuron. You can call [1, 0] to check the weights between first input to second hidden neuron. However, biases are applied to one neuron, so it only has 1 dimension.

