import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.models.surrogate_model import *
from src.data.ff_dataloader import *
from src.preprocessing.transform_into_model_data_ff import *


# Set seeds in order to reproduce results
torch.manual_seed(73)
random.seed(73)
np.random.seed(73)

# Load dataset
# Replace by true labeled dataset if available
train_dataset = DatasetFF(train=True, path="../../data/fitness_function")
test_dataset = DatasetFF(train=False, path="../../data/fitness_function")


# Scale training and test data
scaler = StandardScaler()
scaler.fit(train_dataset.X[:,6:])
train_dataset.scale(scaler)
test_dataset.scale(scaler)


# Build model and switch to GPU if available
fitness_function = FitnessFunction(hidden_sizes=[98, 69])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
fitness_function.to(device)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(fitness_function.parameters(), lr=0.001)


# Data loader
# Useful because it automatically generates batches in the training loop and takes care of shuffling

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Train the model
n_epochs = 1000


# Stuff to store
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for it in range(n_epochs):
    train_loss = []
    for inputs, targets in train_loader:
        # move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = fitness_function(inputs)

        loss = criterion(outputs, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss)

    test_loss = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = fitness_function(inputs)
        loss = criterion(outputs, targets)
        test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss

    if (it + 1) % 1 == 0:
        print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()


# Save trained model

torch.save(fitness_function.state_dict(), "../../models/best_fitness_function.pth")


# Load model and make predictions

model = FitnessFunction(hidden_sizes=[98, 69])
model.load_state_dict(torch.load("../../models/best_fitness_function.pth"))
model.eval()

random_input = torch.from_numpy(np.random.normal(size=[1, 98]).astype(np.float32))

output = model(random_input)
print(output)
print(decode_classes(output))
