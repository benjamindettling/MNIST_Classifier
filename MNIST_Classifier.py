import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
import os
import time

# Select modus: 0 = train from scratch, 1 = load and continue training, 2 = load and evaluate only
modus = 2

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MNIST dataset
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x, start_dim=0))
    ]),
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x, start_dim=0))
    ]),
)

train_loader = DataLoader(train_data, batch_size=64, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=128, pin_memory=True, shuffle=False, drop_last=False)

# Define neural network
class DeepNeuralNet(nn.Module):
    def __init__(self, input_width, hidden_layer_profile, output_width, output_activation=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_layer_profile[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_profile[0], hidden_layer_profile[1]),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_layer_profile[1], output_width)
        self.output_activation = nn.Identity() if not output_activation else output_activation

    def forward(self, input):
        x = self.layers(input)
        output_before_activation = self.output_layer(x)
        return self.output_activation(output_before_activation)

# Training function
def training_loop(dataloader, net, loss_fn, optimiser, verbosity=3, device=device):
    net.train()
    acc_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = net(X)
        loss_val = loss_fn(pred, y)
        acc_loss += loss_val.item()
        optimiser.zero_grad()
        loss_val.backward()
        optimiser.step()
    return acc_loss / len(dataloader)

# Testing function
def testing_loop(dataloader, net, device=device):
    net.eval()
    acc_correct = 0
    acc_count = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            pred_class = pred.argmax(dim=1)
            acc_correct += (pred_class == y).sum().item()
            acc_count += y.size(0)
    accuracy = acc_correct / acc_count
    print(f"Model Accuracy on Test Set: {accuracy:.2%}")
    return accuracy

# Train function
def train(dataloader, net, loss_fn, optimiser, epochs, device=device):
    for t in range(epochs):
        mean_loss = training_loop(dataloader, net, loss_fn, optimiser)
        accuracy = testing_loop(test_loader, net)
        print(f"Epoch {t+1}: Loss {mean_loss:.5f}, Accuracy {accuracy:.2%}")


model = DeepNeuralNet(input_width=28*28, hidden_layer_profile=[512, 256], output_width=10, output_activation=nn.Softmax(dim=1))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Preload model if modus is 1 or 2, or if no saved model exists
if modus in [1, 2] and os.path.exists("mnist_model.pth"):
    print("Loading existing model...")
    checkpoint = torch.load("mnist_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else: print("Train model from scratch...")

# Perform training if modus is 0 or 1
if modus in [0, 1]:
    train(train_loader, model, loss_fn, optimizer, epochs=3)

    # Save the trained model
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "mnist_model.pth")
    print("Model saved as mnist_model.pth")

# Test model
model.eval()
test_accuracy = testing_loop(test_loader, model)
print(f"\nTest Accuracy: {test_accuracy:.2%}")
