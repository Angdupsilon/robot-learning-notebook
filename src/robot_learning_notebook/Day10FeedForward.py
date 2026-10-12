import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameters
input_size = 784 # 28*28 because MNIST image is 28 by 28 pixels, and we flatten it to 1 dimension
hidden_size = 100
num_classes = 10 # 10 classes, digits 0-9
num_epoch = 5 # for now
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset=torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(),download=True)
eval_dataset=torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor())

train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

# Test with Sample
samples = iter(train_loader)
examples = next(samples)

images,labels = examples
print(images.shape, labels.shape) # can't use labels.shape yet because labels are integer

# Check data
for i in range(10):
    pyplot.subplot(2, 5, i+1)
    pyplot.imshow(images[i][0], cmap="gray")
# pyplot.show()

# Model
class NeuralNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # right now tensor 100, 1, 28, 28
        # we need to make it 100, 784 as inputs
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images) # input images, output guess
        loss = criterion(outputs, labels) # evaluate result comparing outputs (predictions) and labels (actual numbers)

        # backwards pass
        optimizer.zero_grad() # clears previous gradient
        loss.backward() # computes new gradient
        optimizer.step()

        if (i+1) %100 == 0:
            print(f"epoch {epoch+1} / {num_epoch}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device) # why are we pushing again?
        labels = labels.to(device)
        outputs = model(images)

        # return value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100*n_correct/n_samples
    print(f"Accuracy = {accuracy}")