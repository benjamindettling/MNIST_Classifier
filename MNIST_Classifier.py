import numpy as np
import os
import requests

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader

import time


train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=Compose([
      ToTensor(),
      Lambda(lambda x: torch.flatten(x, start_dim=0))
    ]),
)
train_loader = DataLoader(train_data, batch_size=64, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepNeuralNet(nn.Module):
  def __init__(self, input_width, hidden_layer_profile, output_width, output_activation=None):
    super().__init__()
    self.layers = nn.ModuleList()

    # create the first hidden layer
    self.layers.append(nn.Linear(input_width, hidden_layer_profile[0]))
    self.layers.append(nn.ReLU())

    # create the internal hidden layers
    for in_width, out_width in zip(hidden_layer_profile[0:-1], hidden_layer_profile[1:]):
      self.layers.append(nn.Linear(in_width, out_width))
      self.layers.append(nn.ReLU())

    self.layers = nn.Sequential(*self.layers)

    # create the output layer
    self.output_layer = nn.Linear(hidden_layer_profile[-1], output_width)
    self.output_activation = nn.Identity() if not output_activation else output_activation

  def forward(self, input):
    x = input

    # loop through the layers to produce the output of the hidden network
    # for layer in self.layers:
      # TODO: pass the intermediate output of the previous layer through the current layer
    x = self.layers(x)  # pass through the sequential list of layers

    # TODO: produce the output of the network from the intermediate output of the last hidden layer
    output_before_activation = self.output_layer(x)

    # TODO: engage the optional activation in self.output_activation on the output_before_activation
    output = self.output_activation(output_before_activation)

    return output

def training_loop(
        dataloader: torch.utils.data.DataLoader,
        net: nn.Module,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        verbosity: int=3,
        device = device):
    size = len(dataloader.dataset)
    last_print_point = 0
    current = 0

    acc_loss = 0
    acc_count = 0
    net.train()
    # for every slice (X, y) of the training dataset
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # perform a forward pass to compute the outputs of the net
        pred = net(X)

        # calculate the loss between the outputs of the net and the desired outputs
        # print(f"pred:{pred.shape} y:{y.shape}")
        # print(y)
        loss_val = loss_fn(pred, y)
        acc_loss += loss_val.item()
        acc_count += 1

        # zero the gradients computed in the previous step
        optimiser.zero_grad()

        # calculate the gradients of the parameters of the net
        loss_val.backward()

        # use the gradients to update the weights of the network
        optimiser.step()

        # compute how many datapoints have already been used for training
        current = batch * len(X)

        # report on the training progress roughly every 10% of the progress
        if verbosity >= 3 and (current - last_print_point) / size >= 0.1:
            loss_val = loss_val.item()
            last_print_point = current
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    return acc_loss / acc_count

def testing_loop(dataloader, net, device = device):
  size = len(dataloader.dataset)
  last_print_point = 0
  current = 0

  acc_correct = 0
  acc_count = 0

  net.eval()
  # for every slice (X, y) of the training dataset
  with torch.no_grad():
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # perform a forward pass to compute the outputs of the net
        pred = net(X)

        # Get the predicted class by selecting the index with the maximum value along the second dimension
        pred_class = pred.argmax(dim=1)

        # compute the number of correct entries
        acc_correct += (pred_class == y).sum().item()
        acc_count += y.size(0)  # Count the number of samples in this batch


  return acc_correct / acc_count

def train(dataloader, net, loss_fn, optimiser, epochs, epoch_frequency=1, device=device, verbosity=3):
  least_loss = None
  if verbosity < 2:
    for t in range(epochs):
      mean_loss = training_loop(dataloader, net, loss_fn, optimiser, verbosity=verbosity)
      accuracy = testing_loop(dataloader, net)
      if not least_loss or mean_loss < least_loss:
        least_loss = mean_loss
  else:
    for t in range(epochs):
        start_time = time.time() # Start the timer
        if verbosity >= 3:
          print(f"Epoch {t+1}\n-------------------------------")

        mean_loss = training_loop(dataloader, net, loss_fn, optimiser, verbosity=verbosity)
        accuracy = testing_loop(dataloader, net)
        if not least_loss or mean_loss < least_loss:
          least_loss = mean_loss

        epoch_time = time.time() - start_time  # Calculate the elapsed time

        if verbosity >= 2 and t%epoch_frequency == 0:
          print(f"Epoch {t:4}: mean loss {mean_loss:.5f}, validation accuracy {accuracy:7.2%}, time: {epoch_time:.2f} seconds")
        if verbosity >= 3:
          print("\n")

  if verbosity >= 1:
    print(f"\nTraining complete, least loss {least_loss}, final validation accuracy {accuracy:.2%}")

  return least_loss

################################################################################

doTrain = True
if doTrain:
  net = DeepNeuralNet(input_width=28*28, hidden_layer_profile=[512, 256], output_width=10, output_activation=nn.Softmax(dim=1))

  training_dataloader = train_loader
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(net.parameters(), lr=0.001)
  net.to(device)
  least_loss = train(training_dataloader, net, loss_fn, optimiser, epochs=15, verbosity=2)

  # Save the trained model
  torch.save(net.state_dict(), "mnist_model.pth")
  print("Model saved as mnist_model.pth")
  model = net
else:
  # Load the trained model
  model = DeepNeuralNet(input_width=28*28, hidden_layer_profile=[512, 256], output_width=10, output_activation=nn.Softmax(dim=1))
  model.load_state_dict(torch.load("mnist_model.pth"))

model.to(device)



test_data = datasets.MNIST(
    root='data',
    train=False,  # This loads the test set
    download=True,
    transform=Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x, start_dim=0))  # Flatten images into vectors
    ]),
)

test_loader = DataLoader(test_data, batch_size=128, pin_memory=True, shuffle=False, drop_last=False)

# test model
model.eval()
test_accuracy = testing_loop(test_loader, model)
print(f"\nTest Accuracy: {test_accuracy:.2%}")
